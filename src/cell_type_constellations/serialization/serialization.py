"""
This module will define functions to write the data needed for an
interactive constellation plot to an HDF5 file
"""

import h5py
import json
import matplotlib
import numpy as np
import pandas as pd
import pathlib
import scipy
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

from cell_type_constellations.cells.cell_set import (
    CellSet
)

from cell_type_constellations.mixture_matrix.mixture_matrix_generator import (
    create_mixture_matrix_from_h5ad,
    create_mixture_matrix_from_coords
)

from cell_type_constellations.visual_elements.fov import (
    FieldOfView
)

import cell_type_constellations.serialization.ingestion as ingestion
import cell_type_constellations.utils.coord_utils as coord_utils
import cell_type_constellations.visual_elements.centroid as centroid
import cell_type_constellations.visual_elements.connection as connection
import cell_type_constellations.hulls.creation as hull_creation
import cell_type_constellations.umap.plot_utils as umap_utils


def serialize_from_h5ad(
        h5ad_path,
        visualization_coords,
        connection_coords_list,
        discrete_fields,
        continuous_fields,
        leaf_field,
        dst_path,
        discrete_color_map=None,
        tmp_dir=None,
        clobber=False,
        k_nn=15,
        n_processors=4,
        fov_height=1080,
        max_radius=35,
        min_radius=5):
    """
    Instantiate and serialize all of the data needed for an interactive
    constellation plot from a single h5ad file.

    Parameters
    ----------
    h5ad_path:
        path to the h5ad file
    visualization_coords:
        a str. The key in obsm where the 2D embedding coordinates
        used for the visualization of the data will be
    connection_coords_list:
        a list of str. Each one is a key in obsm pointing to
        embedding coordinates which will be used to assess whether or
        not two nodes in the constellation plot are connected (can
        be more than 2D, but greater than 2D embeddings can take
        ~ an hour to process)
    discrete_fields:
        a list of str. The columns in obs that refer to discrete
        cell types (these will be nodes in the constellation plot)
    continuous_fields:
        a list of str. The columns in obs corresponding to continuous
        numerical values by which the nodes in the constellation plot
        can be colored
    leaf_field:
        a str. Must be one of discrete_fields. This is the field that
        is considered the leaf level of the taxonomy. This is used
        for constructing the hulls (the contours around discrete_fields
        in visualization space).
    dst_path:
        path to the HDF5 file where the data will be serialized
    discrete_color_map:
        an optional dict mapping discrete_fields to hexadecimal color
        representations. If this is None, a cartoon colormap will
        be created. See notes below.
    tmp_dir:
        path to a directory where scratch files may be written
    clobber:
        a boolean. If False and dst_path exists, crash. If True,
        overwrite
    k_nn:
        an int. The number of nearest neighbors to find for each
        cell when assessing the connectedness of nodes in the
        constellation plot
    n_processors:
        a int. The number of independent worker processes to spin
        up at a time.
    fov_height:
        the height in pixels of the field of view (width will be
        calculated to preserve the aspect ratio of visualization_coords)
    max_radius:
        the maximum radius in pixels that a node in the constellation
        plot can have (approximately)
    min_radius:
        the minimum radius in pixels that a node in the constellation
        plot can have.

    Notes
    -----
    discrete_color_map is structured like
    {
        type_field_1: {
            type_value_1: color1,
            type_value_2: color2,
            type_value_3: color3,
            ...
        },
        type_field_2: {
            type_value4: color4,
            type_value5: color5,
            ...
        },
        ...

    }

    where the type_fields are the values in discrete_fields and
    type_values are the values that those fields can take.

    All type_field, type_value pairs in the h5ad file must be
    represented in the color map.
    """

    t0 = time.time()
    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='constellation_serialization_'
    )
    try:
        _serialize_from_h5ad(
            h5ad_path=h5ad_path,
            visualization_coords=visualization_coords,
            connection_coords_list=connection_coords_list,
            discrete_fields=discrete_fields,
            continuous_fields=continuous_fields,
            leaf_field=leaf_field,
            dst_path=dst_path,
            discrete_color_map=discrete_color_map,
            tmp_dir=tmp_dir,
            k_nn=k_nn,
            n_processors=n_processors,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius,
            clobber=clobber
        )
    finally:
        _clean_up(tmp_dir)
    dur = (time.time()-t0)/60.0
    print(f'SERIALIZATION TOOK {dur:.2e} MINUTES')


def _serialize_from_h5ad(
        h5ad_path,
        visualization_coords,
        connection_coords_list,
        discrete_fields,
        continuous_fields,
        leaf_field,
        dst_path,
        discrete_color_map,
        tmp_dir,
        k_nn,
        n_processors,
        fov_height,
        max_radius,
        min_radius,
        clobber):

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not dst_path.is_file():
            raise RuntimeError(
                f"{dst_path} exists but is not a file"
            )
        if clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_path} exists; run with clobber=True to overwrite"
            )

    visualization_coord_array = coord_utils.get_coords_from_h5ad(
        h5ad_path=h5ad_path,
        coord_key=visualization_coords
    )

    cell_set = CellSet.from_h5ad(
        h5ad_path=h5ad_path,
        discrete_fields=discrete_fields,
        continuous_fields=continuous_fields,
        leaf_field=leaf_field
    )

    if discrete_color_map is None:
        # create a placeholder color map for discrete fields
        # (until we can figure out a consistent way to encode
        # colors associated with discrete_fields)
        discrete_color_map = dict()
        for type_field in cell_set.type_field_list():
            n_values = len(cell_set.type_value_list(type_field))
            mpl_map = matplotlib.colormaps['viridis']
            type_color_map = {
                v: matplotlib.colors.rgb2hex(mpl_map(ii/n_values))
                for ii, v in enumerate(cell_set.type_value_list(type_field))
            }
            discrete_color_map[type_field] = type_color_map

    _validate_discrete_color_map(
        cell_set=cell_set,
        color_map=discrete_color_map
    )

    fov = FieldOfView.from_h5ad(
        h5ad_path=h5ad_path,
        coord_key=visualization_coords,
        fov_height=fov_height,
        max_radius=max_radius,
        min_radius=min_radius
    )

    (embedding_lookup,
     connection_coords_to_mm_path) = get_raw_connection_data_from_h5ad(
         cell_set=cell_set,
         h5ad_path=h5ad_path,
         visualization_coords=visualization_coords,
         connection_coords_list=connection_coords_list,
         k_nn=k_nn,
         n_processors=n_processors,
         tmp_dir=tmp_dir
     )

    serialize_data(
        cell_set=cell_set,
        fov=fov,
        discrete_color_map=discrete_color_map,
        embedding_centroid_lookup=embedding_lookup,
        visualization_coord_array=visualization_coord_array,
        connection_coords_to_mm_path=connection_coords_to_mm_path,
        dst_path=dst_path,
        n_processors=n_processors,
        tmp_dir=tmp_dir
    )

def serialize_from_csv(
        cell_to_cluster_path,
        cluster_annotation_path,
        cluster_membership_path,
        embedding_path,
        hierarchy,
        dst_path,
        tmp_dir=None,
        clobber=False,
        k_nn=15,
        n_processors=4,
        fov_height=1080,
        max_radius=35,
        min_radius=5):
    t0 = time.time()
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        _serialize_from_csv(
            cell_to_cluster_path=cell_to_cluster_path,
            cluster_annotation_path=cluster_annotation_path,
            cluster_membership_path=cluster_membership_path,
            embedding_path=embedding_path,
            hierarchy=hierarchy,
            dst_path=dst_path,
            tmp_dir=tmp_dir,
            clobber=clobber,
            k_nn=k_nn,
            n_processors=n_processors,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius
        )
    finally:
        _clean_up(tmp_dir)
    dur = (time.time()-t0)/60.0
    print(f'SERIALIZATION TOOK {dur:.2e} MINUTES')


def _serialize_from_csv(
        cell_to_cluster_path,
        cluster_annotation_path,
        cluster_membership_path,
        embedding_path,
        hierarchy,
        dst_path,
        tmp_dir=None,
        clobber=False,
        k_nn=15,
        n_processors=4,
        fov_height=1080,
        max_radius=35,
        min_radius=5):

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not dst_path.is_file():
            raise RuntimeError(
                f"{dst_path} exists but is not a file"
            )
        if clobber:
            dst_path.unlink()
        else:
            raise RuntimeError(
                f"{dst_path} exists; run with clobber=True to overwrite"
            )

    ingested_data = ingestion.from_csv(
        cell_to_cluster_path=cell_to_cluster_path,
        cluster_annotation_path=cluster_annotation_path,
        cluster_membership_path=cluster_membership_path,
        embedding_path=embedding_path,
        hierarchy=hierarchy
    )

    cell_set = ingested_data['cell_set']
    embedding_coords = ingested_data['embedding_coords']
    discrete_color_map = ingested_data['discrete_color_map']

    _validate_discrete_color_map(
        cell_set=cell_set,
        color_map=discrete_color_map
    )

    fov = FieldOfView.from_coords(
        coords=embedding_coords,
        fov_height=fov_height,
        max_radius=max_radius,
        min_radius=min_radius
    )

    mm_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='mixture_matrix_',
        suffix='.h5'
    )

    create_mixture_matrix_from_coords(
        cell_set=cell_set,
        coords=embedding_coords,
        k_nn=k_nn,
        dst_path=mm_path,
        tmp_dir=tmp_dir,
        n_processors=n_processors,
        clobber=True,
        chunk_size=100000
    )

    embedding_lookup = centroid.embedding_centroid_lookup_from_coords(
        cell_set=cell_set,
        coords=embedding_coords
    )

    serialize_data(
        cell_set=cell_set,
        fov=fov,
        discrete_color_map=discrete_color_map,
        embedding_centroid_lookup=embedding_lookup,
        visualization_coord_array=embedding_coords,
        connection_coords_to_mm_path={'embedding': mm_path},
        dst_path=dst_path,
        n_processors=n_processors,
        tmp_dir=tmp_dir
    )


def get_raw_connection_data_from_h5ad(
        h5ad_path,
        cell_set,
        visualization_coords,
        connection_coords_list,
        k_nn,
        n_processors,
        tmp_dir):
    """
    Create the embedding space centroids and mixture matrices
    for a constellation plot.

    Parameters
    ----------
    h5ad_path:
        path to the h5ad file
    cell_set:
        the CellSet defining the cells in the constellation plot
    visualization_coords:
        a str. The key in obsm where the 2D embedding coordinates
        used for the visualization of the data will be
    connection_coords_list:
        a list of str. Each one is a key in obsm pointing to
        embedding coordinates which will be used to assess whether or
        not two nodes in the constellation plot are connected (can
        be more than 2D, but greater than 2D embeddings can take
        ~ an hour to process)
    k_nn:
        an int. The number of nearest neighbors to find for each
        cell when assessing the connectedness of nodes in the
        constellation plot
    n_processors:
        a int. The number of independent worker processes to spin
        up at a time.
    tmp_dir:
        path to a directory where scratch files may be written

    Returns
    -------
    embedding_lookup:
        lookup table of embedding space centroids
    centroid_coords_to_mm_path:
        lookup table mapping name of connection coords
        to the path to the HDF5 file where the mixture matrix
        for that coordinate system is stored
    """
    connection_coords_to_mm_path = dict()

    for connection_coords in connection_coords_list:
        print(f'===creating mixture matrices for {connection_coords}')
        mixture_matrix_path = mkstemp_clean(
            dir=tmp_dir,
            prefix=f'{connection_coords}_mixture_matrix_',
            suffix='.h5'
        )
        create_mixture_matrix_from_h5ad(
            cell_set=cell_set,
            h5ad_path=h5ad_path,
            k_nn=k_nn,
            coord_key=connection_coords,
            dst_path=mixture_matrix_path,
            tmp_dir=tmp_dir,
            n_processors=n_processors,
            clobber=True,
            chunk_size=1000000
        )

        connection_coords_to_mm_path[connection_coords] = (
            mixture_matrix_path
        )

    embedding_lookup = centroid.embedding_centroid_lookup_from_h5ad(
        cell_set=cell_set,
        h5ad_path=h5ad_path,
        coord_key=visualization_coords
    )

    return embedding_lookup, connection_coords_to_mm_path


def serialize_data(
        cell_set,
        fov,
        discrete_color_map,
        embedding_centroid_lookup,
        visualization_coord_array,
        connection_coords_to_mm_path,
        dst_path,
        n_processors,
        tmp_dir):
    """
    Parameters
    ----------
    cell_set:
        a CellSet
    fov:
        a FieldOfView
    discrete_color_map:
        dict mapping [type_field][type_value] -> color hex
    embedding_centroid_lookup:
        dict mapping [type_field][type_value] -> EmbeddingSpaceCentroids
    visualization_coord_array:
        (n_cells, 2) array of embedding coords for visualization
    connection_coords_to_mm_path:
        dict mapping key of connection coordinates to mixture matrix hdf5 file
    dst_path:
        path to HDF5 file to be written
    n_processors:
        number of independent worker process to spin up
    tmp_dir:
        directory where scratch files can be written
    """
    centroid_lookup = (
        centroid.pixel_centroid_lookup_from_embedding_centroid_lookup(
            embedding_lookup=embedding_centroid_lookup,
            fov=fov
        )
    )

    discrete_fields = cell_set.type_field_list()
    continuous_fields = cell_set.continuous_field_list()

    fov.to_hdf5(
        hdf5_path=dst_path,
        group_path='fov')

    with h5py.File(dst_path, 'a') as dst:
        dst.create_dataset(
            'discrete_fields',
            data=json.dumps(discrete_fields).encode('utf-8')
        )
        dst.create_dataset(
            'continuous_fields',
            data=json.dumps(continuous_fields).encode('utf-8')
        )
        dst.create_dataset(
            'discrete_color_map',
            data=json.dumps(discrete_color_map).encode('utf-8')
        )

    umap_utils.add_scatterplot_to_hdf5(
        cell_set=cell_set,
        embedding_coords=visualization_coord_array,
        fov=fov,
        discrete_color_map=discrete_color_map,
        hdf5_path=dst_path,
        tmp_dir=tmp_dir
    )

    for type_field in centroid_lookup:
        print(f'===serializing {type_field} connections===')
        centroid.write_pixel_centroids_to_hdf5(
            hdf5_path=dst_path,
            group_path=f'{type_field}/centroids',
            centroid_list=list(centroid_lookup[type_field].values())
        )

        for connection_coords in connection_coords_to_mm_path:
            print(f'======serializing {connection_coords} connections')
            connection_list = connection.get_connection_list(
                pixel_centroid_lookup=centroid_lookup,
                mixture_matrix_file_path=(
                    connection_coords_to_mm_path[connection_coords]
                ),
                type_field=type_field
            )

            connection_list = [conn.to_pixel_space_connection()
                               for conn in connection_list]

            connection.write_pixel_connections_to_hdf5(
                hdf5_path=dst_path,
                group_path=f'{type_field}/connections/{connection_coords}',
                connection_list=connection_list
            )

    hull_creation.create_and_serialize_all_hulls(
        cell_set=cell_set,
        visualization_coords=visualization_coord_array,
        fov=fov,
        dst_path=dst_path,
        n_processors=n_processors,
        tmp_dir=tmp_dir
    )


def _validate_discrete_color_map(color_map, cell_set):
    """
    Validate that all type_field, type_value pairs
    in the cell_set are represented in the color_map
    """
    missing_pairs = []
    for type_field in cell_set.type_field_list():
        if type_field not in color_map:
            missing_pairs.append((type_field, '*'))
            continue
        for type_value in cell_set.type_value_list(type_field):
            if type_value not in color_map[type_field]:
                missing_pairs.append((type_field, type_value))

    if len(missing_pairs) == 0:
        return

    msg = (
        "The following type_field, type_value pairs "
        "were missing from your discrete_color_map:\n"
    )
    for pair in missing_pairs:
        msg += f"{pair}\n"
    raise RuntimeError(msg)


def serialize_graph(
        centroid_lookup,
        mm_path_lookup,
        node_path,
        edge_path):
    """
    Write a graph (in embedding coordinates) to disk.

    Parameters
    ----------
    centroid_lookup:
        dict mapping [level][node] to an EmbeddingSpaceCentroid
    mm_path_lookup:
        dict mapping connection coordintate name to a path
        to an HDF5 file containing the mixture matrix
    node_path:
        path to the CSV file where node information will be written
    edge_path:
        path to the CSV file where edge information will be written
    """

    _serialize_nodes(
        centroid_lookup=centroid_lookup,
        dst_path=node_path)

    _serialize_edges(
        hierarchy=list(centroid_lookup.keys()),
        mm_path_lookup=mm_path_lookup,
        dst_path=edge_path
    )


def _serialize_nodes(centroid_lookup, dst_path):
    node_data = []
    for level in centroid_lookup:
        for node in centroid_lookup[level]:
            centroid = centroid_lookup[level][node]
            node_data.append(
                {"level": level,
                 "node": node,
                 "x": centroid.x,
                 "y": centroid.y,
                 "n_cells": centroid.n_cells
                 }
            )
    pd.DataFrame(node_data).to_csv(dst_path, index=False)


def _serialize_edges(
        hierarchy,
        mm_path_lookup,
        dst_path):
    """
    weight means N of src_node's neighbors pointed to dst_node
    """

    initialized = False
    coord_set_list = list(mm_path_lookup.keys())
    wgt_key_list = []
    for level in hierarchy:
        dataset = dict()
        for coord_set in coord_set_list:
            wgt_key = f'{coord_set}_wgt'
            if level == hierarchy[0]:
                wgt_key_list.append(wgt_key)

            with h5py.File(mm_path_lookup[coord_set], 'r') as src:
                grp = src[level]
                idx_to_label = [
                    v.decode('utf-8') for v in grp['row_key'][()]
                ]
                mm = grp['mixture_matrix'][()]

            nonzero = np.where(mm > 0)
            for i0, i1 in zip(nonzero[0], nonzero[1]):
                if i0 == i1:
                    continue
                key = (level, i0, i1)
                if key not in dataset:
                    dataset[key] = {
                        'level': level,
                        'src_node': idx_to_label[i0],
                        'dst_node': idx_to_label[i1]
                    }
                dataset[key][wgt_key] = mm[i0, i1]

        df = pd.DataFrame(dataset.values())
        df = df[
            ['level', 'src_node', 'dst_node'] + wgt_key_list
        ]
        if initialized:
            header = False
            mode = 'a'
        else:
            header = True
            initialized = True
            mode = 'w'

        df.to_csv(
            dst_path,
            mode=mode,
            header=header,
            index=False)
