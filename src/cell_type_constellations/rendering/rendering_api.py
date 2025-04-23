import h5py
import json
import pathlib

import cell_type_constellations.app.html_utils as html_utils
import cell_type_constellations.visual_elements.centroid as centroid
import cell_type_constellations.visual_elements.connection as connection
import cell_type_constellations.hulls.classes as hull_classes
import cell_type_constellations.visual_elements.fov as fov_utils
import cell_type_constellations.rendering.rendering_utils as rendering_utils


def constellation_svg_from_hdf5(
        hdf5_path,
        centroid_level,
        show_centroid_labels,
        hull_level,
        connection_coords,
        color_by,
        fill_hulls,
        render_metadata=True):

    if hull_level == 'NA':
        hull_level = None

    hull_list = None
    hull_level_list = []

    with h5py.File(hdf5_path, 'r') as src:
        fov = fov_utils.FieldOfView.from_hdf5_handle(
            hdf5_handle=src,
            group_path='fov')

        centroid_list = centroid.read_pixel_centroids_from_hdf5_handle(
            hdf5_handle=src,
            group_path=f'{centroid_level}/centroids')

        connection_list = connection.read_pixel_connections_from_hdf5_handle(
            hdf5_handle=src,
            group_path=f'{centroid_level}/connections/{connection_coords}'
        )

        discrete_field_list = json.loads(
            src['discrete_fields'][()].decode('utf-8')
        )

        continuous_field_list = json.loads(
            src['continuous_fields'][()].decode('utf-8')
        )

        discrete_color_map = json.loads(
           src['discrete_color_map'][()].decode('utf-8')
        )

        connection_coords_list = [
            k for k in src[f'{discrete_field_list[0]}/connections'].keys()
        ]

        if 'hulls' in src.keys():
            hull_level_list = [
                level
                for level in discrete_field_list
                if level in src['hulls'].keys()
            ]
            if hull_level is not None:
                hull_list = []
                for type_value in src['hulls'][hull_level].keys():
                    hull = hull_classes.PixelSpaceHull.from_hdf5_handle(
                            hdf5_handle=src,
                            group_path=f'hulls/{hull_level}/{type_value}'
                        )

                    # somewhat irresponsible patching of hull
                    # to contain type_field and type_value
                    hull.type_field = hull_level
                    hull.type_value = type_value

                    hull_list.append(hull)

    try:
        html = rendering_utils.render_svg(
           fov=fov,
           color_map=discrete_color_map,
           color_by=color_by,
           centroid_list=centroid_list,
           connection_list=connection_list,
           hull_list=hull_list,
           fill_hulls=fill_hulls,
           show_centroid_labels=show_centroid_labels)
    except rendering_utils.CannotColorByError:
        html = f"""
        <p>
        Cannot color {centroid_level} centroids by {color_by};
        perhaps {centroid_level} is a 'parent level' of {color_by}?
        </p>
        """

    if render_metadata:
        taxonomy_name = get_taxonomy_name(hdf5_path)

        html += get_constellation_control_code(
            taxonomy_name=taxonomy_name,
            centroid_level=centroid_level,
            color_by=color_by,
            show_centroid_labels=show_centroid_labels,
            hull_level=hull_level,
            connection_coords=connection_coords,
            fill_hulls=fill_hulls,
            discrete_field_list=discrete_field_list,
            continuous_field_list=continuous_field_list,
            hull_level_list=hull_level_list,
            connection_coords_list=connection_coords_list)

    return html


def get_constellation_control_code(
        taxonomy_name,
        centroid_level,
        show_centroid_labels,
        hull_level,
        color_by,
        connection_coords,
        fill_hulls,
        discrete_field_list,
        continuous_field_list,
        hull_level_list,
        connection_coords_list):

    if hull_level is None:
        hull_level = 'NA'

    default_lookup = {
        'centroid_level': centroid_level,
        'hull_level': hull_level,
        'color_by': color_by,
        'connection_coords': connection_coords
    }

    level_list_lookup = {
        'centroid_level': discrete_field_list,
        'color_by': discrete_field_list + continuous_field_list,
        'hull_level': hull_level_list,
        'connection_coords': connection_coords_list
    }

    html = ""

    html += html_utils.html_front_matter_n_columns(
        n_columns=6)

    html += """<form action="constellation_plot" method="GET">\n"""
    html += f"""<input type="hidden" value="{taxonomy_name}" name="taxonomy_name">\n"""  # noqa: E501
    for i_column, field_id in enumerate(
                                ("centroid_level",
                                 "color_by",
                                 "connection_coords",
                                 "hull_level")):

        default_value = default_lookup[field_id]

        button_name = field_id.replace('_', ' ')

        html += """<div class="column">"""
        html += f"""<fieldset id="{field_id}">\n"""
        html += f"""<label for="{field_id}">{button_name}</label><br>"""  # noqa: E501

        button_values = level_list_lookup[field_id]

        if field_id == 'hull_level':
            button_values.append('NA')

        for level in button_values:
            level_name = level
            html += f"""
            <input type="radio" name="{field_id}" id="{level}" value="{level}" """  # noqa: E501
            if level == default_value:
                html += """checked="checked" """
            html += ">"
            html += f"""
            <label for="{level}">{level_name}</label><br>
            """
        html += """</fieldset>\n"""
        if i_column == 0:
            html += """<input type="submit" value="Reconfigure constellation plot">"""  # noqa: E501
            html += html_utils.end_of_page()

        html += """</div>\n"""

    for field_id, current in (("fill_hulls", fill_hulls),
                              ("show_centroid_labels", show_centroid_labels)):

        button_name = field_id.replace("_", " ")
        if field_id == "show_centroid_labels":
            button_name = "show labels"

        html += """<div class="column">"""
        html += f"""<fieldset id="{field_id}">\n"""
        html += f"""<label for="{field_id}">{button_name}</label><br>"""
        html += f"""<input type="radio" name="{field_id}" id="true" value="true" """  # noqa: E501
        if current:
            html += """checked="checked" """
        html += ">"
        html += """
        <label for="true">True</label><br>
        """
        html += f"""<input type="radio" name="{field_id}" id="false" value="false" """  # noqa: E501
        if not current:
            html += """checked="checked" """
        html += ">"
        html += """
        <label for="false">False</label><br>
        """

        html += """</fieldset></div>\n"""

    html += """
    </form>
    """

    return html


def get_constellation_plot_config(
        data_dir):
    """
    Scan a directory for all .h5 files in that directory.
    Create a dict mapping taxonomy_name to hdf5 path and
    default constellation plot settings. Return that dict.
    """
    data_dir = pathlib.Path(data_dir)

    file_path_list = [n for n in data_dir.rglob('**/*.h5')]
    result = dict()
    for file_path in file_path_list:
        with h5py.File(file_path, 'r') as src:

            taxonomy_name = get_taxonomy_name(
                hdf5_path=file_path
            )

            if taxonomy_name in result:
                raise RuntimeError(
                    f"More than one constellation plot in {data_dir} for "
                    f"taxonomy {taxonomy_name}"
                )

            with h5py.File(file_path, 'r') as src:
                discrete_fields = json.loads(
                    src['discrete_fields'][()].decode('utf-8')
                )

                chosen_field = discrete_fields[-2]
                connection_coords = sorted(
                    src[f'{chosen_field}/connections'].keys()
                )[0]

            this = {
                'path': file_path,
                'centroid_level': discrete_fields[-2],
                'color_by': discrete_fields[-2],
                'hull_level': None,
                'connection_coords': connection_coords
            }

            result[taxonomy_name] = this
    return result


def get_taxonomy_name(hdf5_path):
    file_name = pathlib.Path(hdf5_path).name
    return f'{file_name}'
