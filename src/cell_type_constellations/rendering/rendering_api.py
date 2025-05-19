import base64
import h5py
import json
import pathlib

import cell_type_constellations
import cell_type_constellations.utils.str_utils as str_utils
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
        connection_coords,
        color_by,
        render_metadata=True,
        scatter_plot_level=None,
        enable_download=False):
    scatter_plots = False
    scatter_plot_level_list = []
    with h5py.File(hdf5_path, 'r') as src:
        if 'scatter_plots' in src.keys():
            scatter_plots = True
            raw_scatter_plot_level_set = set(src['scatter_plots'].keys())

    if 'None' in raw_scatter_plot_level_set:
        raw_scatter_plot_level_set.remove('None')

    if scatter_plot_level is None or scatter_plot_level == 'NA':
        scatter_plots = False

    data_packet = load_constellation_data_from_hdf5(
        hdf5_path=hdf5_path,
        centroid_level=centroid_level,
        hull_level=None,
        connection_coords=connection_coords
    )

    fov = data_packet["fov"]
    centroid_list = data_packet["centroid_list"]
    connection_list = data_packet["connection_list"]
    hull_list = data_packet["hull_list"]
    discrete_color_map = data_packet["discrete_color_map"]
    connection_coords_list = data_packet["connection_coords_list"]
    continuous_field_list = data_packet["continuous_field_list"]
    discrete_field_list = data_packet["discrete_field_list"]

    scatter_plot_level_list = [
        d for d in discrete_field_list if d in raw_scatter_plot_level_set
    ]
    scatter_plot_level_list += [
        r for r in raw_scatter_plot_level_set if r not in scatter_plot_level_list
    ]

    html = ""
    if scatter_plots:
        html += html_utils.overlay_style()
        html += """<div class="img-overlay-wrap">"""

        html += get_scatterplot(
            hdf5_path=hdf5_path,
            level=scatter_plot_level,
            fov=fov
        )

    try:
        html += rendering_utils.render_svg(
           fov=fov,
           color_map=discrete_color_map,
           color_by=color_by,
           centroid_list=centroid_list,
           connection_list=connection_list,
           hull_list=[],
           fill_hulls=False,
           show_centroid_labels=show_centroid_labels)

    except rendering_utils.CannotColorByError:
        html = f"""
        <p>
        Cannot color {centroid_level} centroids by {color_by};
        perhaps {centroid_level} is a 'parent level' of {color_by}?
        </p>
        """

    if scatter_plots:
        html += "</div>"

    if render_metadata:
        taxonomy_name = get_taxonomy_name(hdf5_path)

        html += get_constellation_control_code(
            taxonomy_name=taxonomy_name,
            hdf5_path=hdf5_path,
            centroid_level=centroid_level,
            color_by=color_by,
            show_centroid_labels=show_centroid_labels,
            scatter_plot_level=scatter_plot_level,
            connection_coords=connection_coords,
            discrete_field_list=discrete_field_list,
            continuous_field_list=continuous_field_list,
            scatter_plot_level_list=scatter_plot_level_list,
            connection_coords_list=connection_coords_list,
            enable_download=enable_download)

    return html


def load_constellation_data_from_hdf5(
        hdf5_path,
        centroid_level,
        hull_level,
        connection_coords,
        convert_to_embedding=False):

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
            group_path=f'{centroid_level}/centroids',
            fov=fov,
            convert_to_embedding=convert_to_embedding)

        connection_list = connection.read_pixel_connections_from_hdf5_handle(
            hdf5_handle=src,
            group_path=f'{centroid_level}/connections/{connection_coords}',
            fov=fov,
            convert_to_embedding=convert_to_embedding
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
                    group_path=f'hulls/{hull_level}/{type_value}'
                    hull = hull_classes.PixelSpaceHull.from_hdf5_handle(
                            hdf5_handle=src,
                            group_path=group_path,
                            fov=fov,
                            convert_to_embedding=convert_to_embedding
                        )

                    # somewhat irresponsible patching of hull
                    # to contain type_field and type_value
                    hull.type_field = hull_level
                    hull.type_value = str_utils.desanitize_taxon_name(
                        type_value
                    )

                    hull_list.append(hull)

    return {
        "discrete_color_map": discrete_color_map,
        "centroid_list": centroid_list,
        "connection_list": connection_list,
        "hull_list": hull_list,
        "connection_coords_list": connection_coords_list,
        "continuous_field_list": continuous_field_list,
        "discrete_field_list": discrete_field_list,
        "hull_level_list": hull_level_list,
        "fov": fov
    }


def get_constellation_control_code(
        taxonomy_name,
        hdf5_path,
        centroid_level,
        show_centroid_labels,
        scatter_plot_level,
        color_by,
        connection_coords,
        discrete_field_list,
        continuous_field_list,
        scatter_plot_level_list,
        connection_coords_list,
        enable_download=False):

    if scatter_plot_level is None:
        scatter_plot_level = 'NA'

    default_lookup = {
        'centroid_level': centroid_level,
        'scatter_plot_level': scatter_plot_level,
        'color_by': color_by,
        'connection_coords': connection_coords
    }

    level_list_lookup = {
        'centroid_level': discrete_field_list,
        'color_by': discrete_field_list + continuous_field_list,
        'scatter_plot_level': scatter_plot_level_list,
        'connection_coords': connection_coords_list
    }

    html = ""

    html += html_utils.html_front_matter_n_columns(
        n_columns=5)

    html += f"""<p>{taxonomy_name}</p>"""

    if enable_download:
        html += get_download_button(
            hdf5_path=hdf5_path,
            centroid_level=centroid_level,
            color_by=color_by,
            connection_coords=connection_coords,
            scatter_plot_level=scatter_plot_level,
            show_centroid_labels=show_centroid_labels
        )

    html += """<form action="constellation_plot" method="GET">\n"""
    html += f"""<input type="hidden" value="{taxonomy_name}" name="taxonomy_name">\n"""  # noqa: E501
    html += """<div class="row">"""
    html += """<input type="submit" value="Reconfigure constellation plot">"""  # noqa: E501
    html += """</div>"""
    for i_column, field_id in enumerate(
                                ("centroid_level",
                                 "color_by",
                                 "connection_coords",
                                 "scatter_plot_level")):

        html += """<div class="column">"""
        default_value = default_lookup[field_id]

        if field_id == 'scatter_plot_level':
            button_name = 'color UMAP by'
        elif field_id == 'color_by':
            button_name = 'color centroids by'
        else:
            button_name = field_id.replace('_', ' ')

        button_values = level_list_lookup[field_id]

        if field_id == 'scatter_plot_level':
            button_values.append('gray')
            button_values.append('NA')

        if len(button_values) == 1:
            html += f"""<input type="hidden" value="{button_values[0]}" name="{field_id}">\n"""  # noqa: E501
        else:

            html += f"""<fieldset id="{field_id}">\n"""
            html += f"""<label for="{field_id}">{button_name}</label><br>"""  # noqa: E501

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
            html += html_utils.end_of_page()

        html += """</div>\n"""

    for field_id, current in (("show_centroid_labels", show_centroid_labels),):

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
                'scatter_plot_level': 'gray',
                'connection_coords': connection_coords
            }

            result[taxonomy_name] = this
    return result


def get_download_button(
        hdf5_path,
        centroid_level,
        color_by,
        connection_coords,
        scatter_plot_level,
        show_centroid_labels):

    hdf5_path = str(pathlib.Path(hdf5_path).resolve().absolute())

    html = ""
    html += """<form action="download_png" method="GET">\n"""
    html += """<div class="row">"""
    html += """<input type="submit" value="Donwnload png">"""  # noqa: E501
    html += """</div>"""

    html += f"""<input type="hidden" value="{hdf5_path}" name="hdf5_path">\n"""  # noqa: E501
    html += f"""<input type="hidden" value="{centroid_level}" name="centroid_level">\n"""  # noqa: E501
    html += f"""<input type="hidden" value="{color_by}" name="color_by">\n"""  # noqa: E501
    html += f"""<input type="hidden" value="{connection_coords}" name="connection_coords">\n"""  # noqa: E501
    html += f"""<input type="hidden" value="{scatter_plot_level}" name="scatter_plot_level">\n"""  # noqa: E501
    html += f"""<input type="hidden" value="{show_centroid_labels}" name="show_centroid_labels">\n"""  # noqa: E501
    html += """</form>"""

    return html


def get_taxonomy_name(hdf5_path):
    file_name = pathlib.Path(hdf5_path).name
    return f'{file_name}'


def get_scatterplot(
        hdf5_path,
        level,
        fov):
    """
    Return HTML for scatter plot image
    """
    if level is None:
        level = "None"

    with h5py.File(hdf5_path, "r") as src:
        if "scatter_plots" not in src.keys():
            return ""
        if level not in src["scatter_plots"]:
            level = "None"
        data = src[f"scatter_plots/{level}"][()].tobytes()
        data = str(base64.b64encode(data))[2:-1]

    html = f"""
    <img src="data:image/png;base64,{data}" width="{fov.width}px" height="{fov.height}.px">
    """
    return html
