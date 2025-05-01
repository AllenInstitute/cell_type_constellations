import geojson

import cell_type_constellations.visual_elements.centroid as centroid_utils
import cell_type_constellations.visual_elements.connection as connection_utils
import cell_type_constellations.hulls.classes as hull_classes
import cell_type_constellations.plotting.plotting_api as plotting_api



def convert_centroid_to_geojson(
        centroid,
        discrete_color_map):

    if not isinstance(centroid, centroid_utils.PixelSpaceCentroid):
        raise ValueError(
            "centroid must be a PixelSpaceCentroid for geojson conversion; "
            f"you gave {type(centroid)}"
        )
    properties = {
        "radius": centroid.radius,
        "label": centroid.label
    }
    properties['annotations'] = {
        key: {
            'label': centroid.annotation['annotations'][key],
            'color': discrete_color_map[key][centroid.annotation['annotations'][key]]
        }
        for key in centroid.annotation['annotations']
    }
    properties['statistics'] = centroid.annotation['statistics']
    return geojson.Point(
        [centroid.pixel_x, centroid.pixel_y],
        properties=properties
    )


def convert_connection_to_geojson(connection):

    if not isinstance(connection, connection_utils.PixelSpaceConnection):
        raise ValueError(
            "connection must be a PixelSpaceConnection for geojson conversion; "
            f"you gave {type(connection)}"
        )

    data = plotting_api.get_connection_plotting_data(connection)
    properties = {
        "color": data["color"],
        "src": {
            "label": connection.src_label,
            "weight": connection.src_neighbor_fraction
        },
        "dst": {
            "label": connection.dst_label,
            "weight": connection.dst_neighbor_fraction
        }
    }

    return geojson.Polygon(
        [_array_to_list(data["points"])],
        properties=properties
    )


def convert_hull_to_geojson(
        hull,
        discrete_color_map):
    if not isinstance(hull, hull_classes.PixelSpaceHull):
        raise ValueError(
            "hull must be a PixelSpaceHull for geojson conversion; "
            f"you gave {type(hull)}"
        )

    properties = {
        "color": discrete_color_map[hull.type_field][hull.type_value],
        "level": hull.type_field,
        "label": hull.type_value
    }

    polygon_list = []
    for sub_hull in plotting_api.split_up_hull(hull):
        polygon_list.append([_array_to_list(sub_hull)])
    return geojson.MultiPolygon(
        polygon_list,
        properties=properties
    )


def _array_to_list(data):
    return [
        list(r) for r in data
    ]
