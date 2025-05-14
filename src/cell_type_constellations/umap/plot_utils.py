"""
Scripts to create the UMAP scatter plot in temporary files
"""

import matplotlib.figure
import numpy as np
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
    
