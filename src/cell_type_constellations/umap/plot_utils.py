"""
Scripts to create the UMAP scatter plot in temporary files
"""

import h5py
import matplotlib.figure
import numpy as np
import pathlib
import PIL.ImageColor

from cell_type_mapper.utils.utils import mkstemp_clean


def plot_embedding(
        cell_set,
        embedding_coords,
        fov,
        discrete_color_map,
        color_by,
        dst_path):
    """
    Parameters
    ----------
    cell_set:
        the CellSet containing the per-cell metadata
    embedding_coords:
        the (n_cells, 2) numpy array of embedding coordinates
        (rows must be in the same order as the cells in cell_set)
    fov:
        the FieldOfView defining the visualization
    discrete_color_map:
        the dict mapping [type_field][type_value] to color
    color_by:
        the type_field we are coloring by
        (if None, everything will be gray)
    dst_path:
        path to file where scatter plot will be saved
    """
    alpha = 0.75
    rng = np.random.default_rng(771812311)
    gray = '#dddddd'
    n_cells = embedding_coords.shape[0]
    if color_by is None:
        color_array = [gray]*n_cells
    else:
        type_value_array = cell_set.type_value_from_idx(
            type_field=color_by,
            idx_array=np.arange(n_cells, dtype=int)
        )
        color_array = [
            discrete_color_map[color_by][value]
            for value in type_value_array
        ]
        color_array = []
        faded_color = dict()
        for value in type_value_array:
            if value not in faded_color:
                orig = PIL.ImageColor.getcolor(
                    discrete_color_map[color_by][value],
                    'RGB'
                )
                orig = [o/255.0 for o in orig]
                new_color = [
                    alpha*o+(1.0-alpha)
                    for o in orig
                ]
                faded_color[value] = tuple(new_color)
            color_array.append(faded_color[value])

    color_array = np.array(color_array)

    embedding_coords = fov.transform_to_pixel_coordinates(
        embedding_coords=embedding_coords
    )

    xx = embedding_coords[:, 0]
    yy = embedding_coords[:, 1]

    # convert to array coordinates for matplotlib (?)
    yy = fov.height - yy

    idx = np.arange(n_cells, dtype=int)
    rng.shuffle(idx)
    xx = xx[idx]
    yy = yy[idx]
    color_array = color_array[idx]
    fig = matplotlib.figure.Figure(figsize=(20, 20*fov.height/fov.width))
    axis = fig.add_subplot(1, 1, 1)
    axis.scatter(
        xx,
        yy,
        c=color_array,
        s=1
    )
    axis.axis('off')
    axis.set_xlim((0, fov.width))
    axis.set_ylim((0, fov.height))
    fig.tight_layout()
    fig.savefig(dst_path, bbox_inches=0)


def add_scatterplot_to_hdf5(
        cell_set,
        embedding_coords,
        fov,
        discrete_color_map,
        hdf5_path,
        tmp_dir):
    """
    Add scatter plots to HDF5 serialization of
    constellation plot.

    Parameters
    ----------
    cell_set:
        the CellSet containing the per-cell metadata
    embedding_coords:
        the (n_cells, 2) numpy array of embedding coordinates
        (rows must be in the same order as the cells in cell_set)
    fov:
        the FieldOfView defining the visualization
    discrete_color_map:
        the dict mapping [type_field][type_value] to color
    color_by:
        the type_field we are coloring by
        (if None, everything will be gray)
    hdf5_path:
       path to hdf5 file where data is being stored
    tmp_dir:
        directory where we can write temporary png files
    """
    print("SERIALIZING SCATTERPLOTS")
    tmp_dir = pathlib.Path(tmp_dir)
    color_by_list = cell_set.type_field_list() + [None]
    for color_by in color_by_list:
        print(f"    SERIALIZED SCATTERPLOT COLORED BY {color_by}")
        png_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir,
                prefix=str(color_by)+'_',
                suffix='.png'
            )
        )

        try:
            plot_embedding(
                cell_set=cell_set,
                embedding_coords=embedding_coords,
                fov=fov,
                discrete_color_map=discrete_color_map,
                color_by=color_by,
                dst_path=png_path
            )
            with h5py.File(hdf5_path, 'a') as dst:
                with open(png_path, 'rb') as src:
                    data = np.void(src.read())
                dst.create_dataset(
                    f'scatter_plots/{str(color_by)}',
                    data=data
                )
        finally:
            png_path.unlink()
