import matplotlib

import cell_type_constellations.rendering.rendering_api as rendering_api


def plot_constellation_in_mpl(
        hdf5_path,
        centroid_level,
        hull_level,
        color_by_level,
        axis,
        connection_coords='X_umap',
        zorder_base=1):

    hull_zorder = zorder_base
    connection_zorder = hull_zorder + 1
    centroid_zorder = connection_zorder + 1

    constellation_data = rendering_api.load_constellation_data_from_hdf5(
        hdf5_path=hdf5_path,
        centroid_level=centroid_level,
        hull_level=hull_level,
        connection_coords=connection_coords,
        convert_to_embedding=True
    )

    color_map = constellation_data['discrete_color_map']

    for centroid in constellation_data['centroid_list']:
        color_key = centroid.annotation['annotations'][color_by_level]
        color = color_map[color_by_level][color_key]
        node = matplotlib.patches.Circle(
            (centroid.pixel_x, centroid.pixel_y),
            radius=centroid.radius,
            facecolor=color,
            edgecolor='#aaaaaa'
        )
        axis.add_artist(node)
