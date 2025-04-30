import h5py
import multiprocessing
import numpy as np
import pathlib
import scipy
import tempfile
import time

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list
)

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

import cell_type_constellations.utils.coord_utils as coord_utils


def create_mixture_matrix_from_h5ad(
        cell_set,
        h5ad_path,
        k_nn,
        coord_key,
        dst_path,
        tmp_dir,
        n_processors,
        clobber=False,
        chunk_size=100000):
    """
    Find the mixture matrices needed to assess the connectivity
    of nodes in a constellation plot. Store the results in an
    HDF5 file.

    Data for the mixture matrices is taken from a single
    h5ad file.

    Parameters
    ----------
    cell_set:
        a CellSet as defined in cells/cell_set.py
    h5ad_path:
        path to the h5ad file containing the latent space
    k_nn:
        number of nearest neighbors to query for each point
    coord_key:
        the key in obsm that points to the array containing the
        latent space coordinates
    dst_path:
        path to the h5ad file where the mixture matrices
        will be saved
    tmp_dir:
        path to scratch directory where temporary files
        can be written
    n_processors:
        number of independent worker processes to spin up
    clobber:
        a boolean. If False and dst_path already exists,
        raise an exception. If True, overwrite.
    chunk_size:
        the number of cells to process in a single worker
        process

    Returns
    -------
    None
        Mixture matrices for all type_fields in the cell_set
        are saved in the h5ad file at dst_path
    """
    kd_tree = _get_kd_tree_from_h5ad(
        h5ad_path=h5ad_path,
        coord_key=coord_key)

    create_mixture_matrix(
        cell_set=cell_set,
        kd_tree=kd_tree,
        k_nn=k_nn,
        dst_path=dst_path,
        clobber=clobber,
        tmp_dir=tmp_dir,
        n_processors=n_processors,
        chunk_size=chunk_size)


def _get_kd_tree_from_h5ad(
        h5ad_path,
        coord_key):
    """
    Extract a set of coordinates from obsm in and h5ad file.
    Convert them into a KD Tree and return

    Parameters
    ----------
    h5ad_path:
        the path to the h5ad file
    coord_key:
        the key (within obsm) of the coordinate array being extracted

    Returns
    -------
    kd_tree:
        a scipy.spatial.cKDTree built off of the corresponding
        coordinates
    """
    coords = coord_utils.get_coords_from_h5ad(
        h5ad_path,
        coord_key=coord_key
    )
    return scipy.spatial.cKDTree(coords)


def create_mixture_matrix_from_coords(
        cell_set,
        coords,
        k_nn,
        dst_path,
        tmp_dir,
        n_processors,
        clobber=False,
        chunk_size=100000):

    kd_tree = scipy.spatial.cKDTree(coords)
    create_mixture_matrix(
        cell_set=cell_set,
        kd_tree=kd_tree,
        k_nn=k_nn,
        dst_path=dst_path,
        tmp_dir=tmp_dir,
        n_processors=n_processors,
        clobber=clobber,
        chunk_size=chunk_size
    )


def create_mixture_matrix(
        cell_set,
        kd_tree,
        k_nn,
        dst_path,
        tmp_dir,
        n_processors,
        clobber=False,
        chunk_size=100000):
    """
    Parameters
    ----------
    cell_set:
        a CellSet as defined in cells/cell_set.py
    kd_tree:
        a KD Tree built from the latent variables
        defining the space in which connection strength
        is evaluated
    k_nn:
        number of nearest neighbors to query for each point
    dst_path:
        path to the h5ad file where the mixture matrices
        will be saved
    tmp_dir:
        path to scratch directory where temporary files
        can be written
    n_processors:
        number of independent worker processes to spin up
    clobber:
        a boolean. If False and dst_path already exists,
        raise an exception. If True, overwrite.
    chunk_size:
        the number of cells to process in a single worker
        process

    Returns
    -------
    None
        Mixture matrices for all type_fields in the cell_set
        are saved in the h5ad file at dst_path
    """

    n_dim = kd_tree.data.shape[1]
    factor = np.ceil(n_dim/2).astype(int)

    chunk_size = max(
        5000,
        min(
            chunk_size,
            np.ceil(cell_set.n_cells/(factor*n_processors)).astype(int)
        )
    )

    dst_path = pathlib.Path(dst_path)
    if dst_path.exists():
        if not clobber:
            raise RuntimeError(
                f"{dst_path} already exists"
            )
        if not dst_path.is_file():
            raise RuntimeError(
                f"{dst_path} already exists, but is not a file"
            )
        dst_path.unlink()

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='mixture_matrix_calculation_'
    )

    try:
        _create_mixture_matrix(
            cell_set=cell_set,
            kd_tree=kd_tree,
            k_nn=k_nn,
            dst_path=dst_path,
            tmp_dir=tmp_dir,
            n_processors=n_processors,
            chunk_size=chunk_size)
    finally:
        _clean_up(tmp_dir)


def _create_mixture_matrix(
        cell_set,
        kd_tree,
        k_nn,
        dst_path,
        tmp_dir,
        n_processors,
        chunk_size):

    n_cells = kd_tree.data.shape[0]
    tmp_path_list = []
    process_list = []
    for i0 in range(0, n_cells, chunk_size):
        i1 = min(n_cells, i0+chunk_size)
        chunk = np.arange(i0, i1, dtype=int)
        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            prefix=f'mixture_matrix_{i0}_{i1}_',
            suffix='.h5'
        )
        tmp_path_list.append(tmp_path)
        p = multiprocessing.Process(
            target=_create_sub_mixture_matrix,
            kwargs={
                'cell_set': cell_set,
                'kd_tree': kd_tree,
                'subset_idx': chunk,
                'k_nn': k_nn,
                'dst_path': tmp_path
            }
        )
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    # join tmp files
    row_key_reference = dict()
    with h5py.File(dst_path, 'w') as dst:
        dst.create_dataset('k_nn', data=k_nn)
        for type_field in cell_set.type_field_list():
            n_types = len(cell_set.type_value_list(type_field))
            grp = dst.create_group(type_field)
            grp.create_dataset(
                'mixture_matrix',
                shape=(n_types, n_types),
                dtype=int
            )
            row_key = np.array(
                [val.encode('utf-8')
                 for val in cell_set.type_value_list(type_field)]
            )
            grp.create_dataset(
                'row_key',
                data=row_key
            )
            row_key_reference[type_field] = row_key
        for tmp_path in tmp_path_list:
            with h5py.File(tmp_path, 'r') as src:
                for type_field in cell_set.type_field_list():

                    # make sure that all the sub matrices
                    # have the same name-to-index column
                    # and row mappings
                    np.testing.assert_array_equal(
                        src[type_field]['row_key'][()],
                        row_key_reference[type_field]
                    )

                    dst[type_field]['mixture_matrix'][:, :] += (
                        src[type_field]['mixture_matrix'][()]
                    )


def _create_sub_mixture_matrix(
        cell_set,
        kd_tree,
        subset_idx,
        k_nn,
        dst_path):

    t0 = time.time()
    matrix_lookup = dict()
    for type_field in cell_set.type_field_list():
        type_value_list = cell_set.type_value_list(type_field)
        n_value = len(type_value_list)
        matrix_lookup[type_field] = np.zeros((n_value, n_value), dtype=int)

    # the k=k_nn+1 is because each cell's nearest neighbor will
    # always be itself.
    neighbors = kd_tree.query(
        x=kd_tree.data[subset_idx, :],
        k=k_nn+1
    )[1][:, 1:]

    rowcol_lookup = dict()

    for type_field in cell_set.type_field_list():

        type_value_list = cell_set.type_value_list(type_field)
        type_value_to_idx = {
            v: ii for ii, v in enumerate(type_value_list)
        }
        rowcol_lookup[type_field] = np.array(
            [val.encode('utf-8') for val in type_value_list]
        )

        row_values = cell_set.type_value_from_idx(
            type_field=type_field,
            idx_array=subset_idx)

        row_idx_array = np.array(
            [type_value_to_idx[v] for v in row_values]
        )

        for ii, row_idx in enumerate(row_idx_array):

            col_values = cell_set.type_value_from_idx(
                type_field=type_field,
                idx_array=neighbors[ii, :]
            )

            col_idx_array = np.array(
                [type_value_to_idx[v] for v in col_values]
            )

            unq_arr, ct_arr = np.unique(col_idx_array, return_counts=True)
            matrix_lookup[type_field][row_idx, unq_arr] += ct_arr

    with h5py.File(dst_path, 'w') as dst:
        for type_field in cell_set.type_field_list():
            grp = dst.create_group(type_field)
            grp.create_dataset(
                'mixture_matrix',
                data=matrix_lookup[type_field]
            )
            grp.create_dataset(
                'row_key',
                data=rowcol_lookup[type_field]
            )

    dur = (time.time()-t0)/60.0
    print(f'=======finished one chunk of {len(subset_idx)} cells in '
          f'{dur:.2e} minutes=======')
