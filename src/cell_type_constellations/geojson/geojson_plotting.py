import geojson
import matplotlib
import numpy as np

import cell_type_constellations.utils.geometry_utils as geometry_utils


def plot_geojson_constellation(
        geojson_path,
        axis,
        plot_hulls=True,
        zorder=0):
    """
    Plot a constellation plot from a geojson file.

    Parameters
    ----------
    geojson_path:
        the path to the file containing the geojson
        encoding of the constellation plot
    axis:
        the matplotlib axis on which to plot the constellation
        plot
    plot_hulls:
        boolean indicating whether or not to plot the hulls
        containing the cell types in UMAP space
    zorder:
        zorder of the constellation plot's lowest layer
    """
    with open(geojson_path, 'rb') as src:
        geojson_data = geojson.load(src)

    hull_zorder = zorder
    connection_zorder = zorder+1
    centroid_zorder = zorder+2
    centroid_level = geojson_data.properties['centroid_level']

    for feature in geojson_data.features:
        if feature.geometry.type == 'Point':
            plot_geojson_centroid(
                geojson_centroid=feature.geometry,
                axis=axis,
                zorder=centroid_zorder,
                color_by_level=centroid_level
            )
        elif feature.geometry.type == 'Polygon':
            plot_geojson_polygon(
                axis=axis,
                polygon=feature.geometry,
                zorder=connection_zorder,
                fill=True
            )
        elif feature.geometry.type == 'MultiPolygon':
            if plot_hulls:
                plot_geojson_multipolygon(
                    axis=axis,
                    multi_polygon=feature.geometry,
                    zorder=hull_zorder,
                    fill=False
                )
        else:
            raise NotImplementedError(
                "Unclear how to plot feature of type "
                f"{feature.geometry.type}"
            )


def plot_geojson_centroid(
        geojson_centroid,
        axis,
        zorder,
        color_by_level):
    """
    Plot a node centroid on a matplotlib axis.

    Parameters
    ----------
    geojson_centroid:
         the geojson.Point representing the node centroid
    axis:
        the matplotlib axis in which to plot the centroid
    zorder:
        the zorder at which to plot the centroid
    color_by_level:
        a string indicating the level by which to color the
        centroids (it is, for instance, possible to plot
        the subclass centroids but color them by the class
        they belong to)
    """
    color = geojson_centroid.properties['annotations'][color_by_level]['color']
    radius = geojson_centroid.properties['radius']
    node = matplotlib.patches.Circle(
        (geojson_centroid.coordinates[0],
         geojson_centroid.coordinates[1]),
        radius=radius,
        facecolor=color,
        edgecolor='#bbbbbb',
        zorder=zorder
    )
    axis.add_artist(node)


def plot_geojson_polygon(
        axis,
        polygon,
        zorder,
        fill=False):
    """
    Plot a geojson polygon on a matplotlib axis

    Parameters
    ----------
    axis:
        the matplotlib axis on which to plot the polygon
    polygon:
        the geojson.Polygon
    zorder:
        the zorder to apply to the polygon
    fill:
        a boolean indicating whether or not to fill the polygon
    """
    color = polygon.properties['color']
    plot_geojson_polygon_coords(
        axis=axis,
        polygon_coords=polygon.coordinates,
        color=color,
        zorder=zorder,
        fill=fill
    )


def plot_geojson_multipolygon(
        axis,
        multi_polygon,
        zorder,
        fill=False):
    """
    Plot a geojson MultiPolygon on a matplotlib axis

    Parameters
    ----------
    axis:
        the matplotlib axis on which to plot the polygon
    multi_polygon:
        the geojson.MultiPolygon
    zorder:
        the zorder to apply to the MultiPolygon
    fill:
        a boolean indicating whether or not to fill the polygon
    """
    color = multi_polygon.properties['color']
    for sub_coords in multi_polygon.coordinates:
        plot_geojson_polygon_coords(
            axis=axis,
            polygon_coords=sub_coords,
            color=color,
            zorder=zorder,
            fill=fill
        )


def plot_geojson_polygon_coords(
        axis,
        polygon_coords,
        color,
        zorder,
        fill=False):
    """
    Plot a geojson polygon on a matplotlib axis.

    Parameters
    ----------
    axis:
        The matplotlib axis on which to plot the polygon
    polygon_coords:
        The coordinates of a geojson Polygon
        (can also be a single set of coordinate from a MultiPolygon)
    color:
        The color to apply to the polygon
    zorder:
        The zorder to apply when plotting the polygon
    fill:
        a boolean indicating whether or not to fill the polygon
    """
    if len(polygon_coords) not in (1, 2):
        raise ValueError(
            f"Polygon coordinates specify {len(polygon_coords)} lists "
            "of points; must be either 1 or 2"
        )

    polygon_coords = np.array(polygon_coords)
    if polygon_coords.shape[0] == 2:
        outer = coords[0, :]
        inner = coords[1, :]
        if fill:
            # make outer polygon counter clockwise and inner polygon clockwise
            outer_cp = geometry_utils.cross_product_2d(
                outer[1, :]-outer[0, :], outer[2, :]-outer[1, :]
            )
            if outder_cp[2] < 0.0:
                outer = outer[-1::-1, :]

            inner_cp = geometry_utils.cross_product_2d(
                inner[1, :]-inner[0, :], inner[2, :]-inner[1, :]
            )
            if inner_cp[2] > 0.0:
                inner = inner[-1::-1, :]

            x_arr = np.concatenate([outer[:, 0], inner[:, 0]])
            y_arr = np.concatenate([outer[:, 1], inner[:, 1]])

    else:
        if fill:
            x_arr = polygon_coords[0][:, 0]
            y_arr = polygon_coords[0][:, 1]
        outer = polygon_coords[0]
        inner = None

    if fill:
        axis.fill(
            x_arr,
            y_arr,
            c=color,
            linewidth=2,
            zorder=zorder
        )
    else:
        axis.plot(
            outer[:, 0],
            outer[:, 1],
            color=color,
            linewidth=2,
            zorder=zorder
        )
        if inner is not None:
            axis.plot(
                inner[:, 0],
                inner[:, 1],
                color=color,
                linewidth=2,
                zorder=zorder
            )
