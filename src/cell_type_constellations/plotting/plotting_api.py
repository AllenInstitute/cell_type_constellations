import matplotlib
import numpy as np

import cell_type_constellations.rendering.rendering_api as rendering_api
import cell_type_constellations.plotting.bezier as bezier_utils


def plot_constellation_in_mpl(
        hdf5_path,
        centroid_level,
        hull_level,
        color_by_level,
        axis,
        connection_coords='X_umap',
        zorder_base=1,
        fill_hulls=False,
        show_labels=False):

    fontsize = 15
    hull_zorder = zorder_base
    connection_zorder = hull_zorder + 2
    centroid_zorder = connection_zorder + 2

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
            edgecolor='#bbbbbb',
            zorder=centroid_zorder
        )
        axis.add_artist(node)
        if show_labels:
            axis.text(
                centroid.pixel_x,
                centroid.pixel_y,
                s=centroid.annotation['annotations'][centroid_level],
                fontsize=fontsize,
                color='#555555',
                zorder=centroid_zorder+1
            )

    for connection in constellation_data['connection_list']:
        plot_connection_in_mpl(
            connection=connection,
            axis=axis,
            zorder=connection_zorder
        )

    if hull_level is not None:
        for hull in constellation_data['hull_list']:
            plot_hull_in_mpl(
                hull=hull,
                color=color_map[hull.type_field][hull.type_value],
                axis=axis,
                fill=fill_hulls,
                zorder=hull_zorder
            )


def plot_connection_in_mpl(connection, axis, zorder):
    plotting_data = get_connection_plotting_data(connection)
    axis.fill(
        plotting_data['points'][:, 0],
        plotting_data['points'][:, 1],
        color=plotting_data['color'],
        zorder=zorder)


def get_connection_plotting_data(connection):

    corner_pts = connection.rendering_corners
    ctrl_pts = connection.bezier_control_points

    bez01 = bezier_utils.quadratic_bezier(
        src_pt=corner_pts[0, :],
        dst_pt=corner_pts[1, :],
        ctrl_pt=ctrl_pts[0, :]
    )

    bez23 = bezier_utils.quadratic_bezier(
        src_pt=corner_pts[2, :],
        dst_pt=corner_pts[3, :],
        ctrl_pt=ctrl_pts[1, :]
    )

    pts = np.vstack(
        [corner_pts[0, :],
         bez01,
         corner_pts[1:2, :],
         bez23,
         corner_pts[3, :],
         corner_pts[0, :]
         ]
    )

    return {
        'points': pts,
        'color': '#bbbbbb'
    }


def plot_hull_in_mpl(
                hull,
                color,
                axis,
                fill=False,
                zorder=0):

    sub_hull_list = split_up_hull(hull)
    for sub_hull in sub_hull_list:
        if fill:
            axis.fill(
                sub_hull[:, 0],
                sub_hull[:, 1],
                c=color,
                zorder=zorder,
                alpha=0.5,
                linewidth=2
            )
        else:
            axis.plot(
                sub_hull[:, 0],
                sub_hull[:, 1],
                c=color,
                zorder=zorder,
                linewidth=2
            )


def split_up_hull(hull):
    """
    Return list of (N, 2) arrays of points on hull
    """
    result = []
    for idx in range(hull.n_sub_hulls):
        sub_hull = hull[idx]
        this = np.array(
            [pt for pt in sub_hull[::4]] + [sub_hull[0, :]]
        )
        result.append(this)
    return result
