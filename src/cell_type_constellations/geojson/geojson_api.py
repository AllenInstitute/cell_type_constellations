import geojson
import json
import pathlib

import cell_type_constellations.geojson.geojson_utils as geojson_utils
import cell_type_constellations.rendering.rendering_api as rendering_api


def convert_constellation_to_geojson(
        src_hdf5_path,
        dst_path,
        centroid_level,
        hull_level,
        connection_coords='embedding',
        clobber=False):

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

    data = rendering_api.load_constellation_data_from_hdf5(
        hdf5_path=src_hdf5_path,
        centroid_level=centroid_level,
        hull_level=hull_level,
        connection_coords=connection_coords,
        convert_to_embedding=True
    )

    properties = {
        'centroid_level': centroid_level,
        'hull_level': hull_level,
        'connection_coords': connection_coords
    }

    discrete_color_map = data['discrete_color_map']

    feature_list = []
    for centroid in data['centroid_list']:
        feature_list.append(
            geojson.Feature(
                geometry=geojson_utils.convert_centroid_to_geojson(
                    centroid=centroid,
                    discrete_color_map=discrete_color_map
                ),
                properties={
                    "class": "centroid"
                }
            )
        )

    for connection in data['connection_list']:
        feature_list.append(
            geojson.Feature(
                geometry=geojson_utils.convert_connection_to_geojson(
                    connection
                ),
                properties={
                    "class": "connection"
                }
            )
        )

    if data['hull_list'] is not None:
        for hull in data['hull_list']:
            feature_list.append(
                geojson.Feature(
                    geometry=geojson_utils.convert_hull_to_geojson(
                        hull=hull,
                        discrete_color_map=discrete_color_map
                    )
                )
            )

    feature_collection = geojson.FeatureCollection(
        feature_list,
        properties=properties
    )
    error_msg = feature_collection.errors()
    if len(error_msg) > 0:
        raise RuntimeError(json.dumps(error_msg, indent=2))
    with open(dst_path, 'w') as dst:
        dst.write(geojson.dumps(feature_collection, indent=2))
