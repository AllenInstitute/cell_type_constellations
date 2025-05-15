import numpy as np

from cell_type_constellations.visual_elements.centroid import (
   PixelSpaceCentroid
)

from cell_type_constellations.visual_elements.connection import (
    PixelSpaceConnection
)

from cell_type_constellations.rendering.continuous_color_map import (
    ContinuousColorMap,
    get_colorbar_code
)

import cell_type_constellations.rendering.hull_rendering as hull_rendering


class CannotColorByError(Exception):
    pass


def render_svg(
        fov,
        color_map,
        color_by,
        centroid_list,
        connection_list=None,
        hull_list=None,
        fill_hulls=False,
        show_centroid_labels=True):

    if connection_list is not None:
        connection_code = render_connection_list(connection_list)
    else:
        connection_code = ''

    hull_code = ""
    if hull_list is not None:
        for hull in hull_list:
            hull_code += hull_rendering.render_hull(
                hull=hull,
                color=color_map[hull.type_field][hull.type_value],
                type_field=hull.type_field,
                type_value=hull.type_value,
                fill=fill_hulls
            )

    (centroid_code,
     fov) = render_centroid_list(
        centroid_list=centroid_list,
        color_map=color_map,
        color_by=color_by,
        fov=fov,
        show_label=show_centroid_labels)

    code = get_svg_header(fov)
    code += hull_code
    code += connection_code
    code += centroid_code
    code += "</svg>\n"
    return code


def get_svg_header(fov):
    result = (
            f'<svg height="{fov.height}px" width="{fov.width}px" '
            'xmlns="http://www.w3.org/2000/svg">\n'
        )
    return result


def render_centroid_list(
        centroid_list,
        color_map,
        color_by,
        fov,
        show_label=True):
    """
    Returns the SVG code for the centroids and fov
    (in case fov needs to be updated to accommodate
    a colorbar or some other legend)
    """
    centroid_code = ""

    if color_by in centroid_list[0].annotation['statistics']:
        color_map = ContinuousColorMap(
            centroid_list=centroid_list,
            color_by=color_by
        )
        (centroid_code,
         fov) = get_colorbar_code(color_map=color_map, fov=fov)

    for el in centroid_list:
        centroid_code += render_centroid(
            centroid=el,
            color_map=color_map,
            color_by=color_by,
            show_label=show_label)

    return centroid_code, fov


def render_centroid(
        centroid,
        color_map,
        color_by,
        show_label=True):

    fontsize = 10

    is_stats = False
    if isinstance(color_map, ContinuousColorMap):
        color_value = centroid.annotation['statistics'][color_by]['mean']
        color = color_map.value_to_rgb(color_value)
        is_stats = True
    else:
        if color_by not in centroid.annotation['annotations']:
            raise CannotColorByError(
                f"Cannot color centroid {centroid.label} by "
                f"column {color_by}; not present in annotation"
            )
        color_value = centroid.annotation['annotations'][color_by]
        color = color_map[color_by][color_value]

    if not isinstance(centroid, PixelSpaceCentroid):
        raise RuntimeError(
            "Can only render PixelSpaceCentroid; your centroid "
            f"is of type {type(centroid)}"
        )

    hover_msg = (
        f"{centroid.label} -- {centroid.n_cells:.2e} cells"
    )
    if is_stats:
        mu = centroid.annotation['statistics'][color_by]['mean']
        std = np.sqrt(centroid.annotation['statistics'][color_by]['var'])
        hover_msg += f"\n{color_by}: {mu:.2e} +/- {std:.2e}"
    else:
        color_label = centroid.annotation['annotations'][color_by]
        if color_label != centroid.label:
            hover_msg += f"\n{color_by}: {color_label}"

    result = """    <a>\n"""

    result += (
        f"""        <circle r="{centroid.radius}px" cx="{centroid.pixel_x}px" cy="{centroid.pixel_y}px" """  # noqa: E501
        f"""fill="{color}" stroke="#bbbbbb" stroke-width="0.25"/>\n"""
    )
    result += """        <title>\n"""
    result += f"""{hover_msg}\n"""
    result += """        </title>\n"""

    result += "    </a>\n"

    if show_label:
        short_label = centroid.label.split(':')[-1]
        first_param = short_label.split('_')[0]
        try:
            display_label = int(first_param)
        except Exception:
            display_label = short_label
        label_code = f"""
        <text x="{centroid.pixel_x}px" y="{centroid.pixel_y}"
        font-size="{fontsize}" stroke="#DDDDDD"
        fill="#000000" stroke-width="0.3px">
        {display_label}
        </text>\n
        """
        result += label_code

    return result


def render_connection_list(connection_list):
    connection_code = ""
    for conn in connection_list:
        connection_code += render_connection(conn)

    print(f'n_conn {len(connection_list)}')
    return connection_code


def render_connection(this_connection):

    if not isinstance(this_connection, PixelSpaceConnection):
        raise RuntimeError(
            "Can only render instances of PixelSpaceConnection; "
            f"your connection is of type {type(this_connection)}"
        )

    title = (
        f"{this_connection.src_label} "
        f"({this_connection.src_neighbor_fraction:.2e} of neighbors) "
        "-> "
        f"{this_connection.dst_label} "
        f"({this_connection.dst_neighbor_fraction:.2e} of neighbors)"
    )

    pts = this_connection.rendering_corners
    ctrl = this_connection.bezier_control_points

    result = """    <a>\n"""
    result += "        <path "
    result += f"""d="M {pts[0][0]} {pts[0][1]} """
    result += get_bezier_curve(
                src=pts[0],
                dst=pts[1],
                ctrl=ctrl[0])
    result += f"L {pts[2][0]} {pts[2][1]} "
    result += get_bezier_curve(
                src=pts[2],
                dst=pts[3],
                ctrl=ctrl[1])
    result += f"""L {pts[0][0]} {pts[0][1]}" """
    result += """stroke="transparent" fill="#bbbbbb"/>\n"""
    result += "        <title>\n"
    result += f"        {title}\n"
    result += "        </title>\n"
    result += "    </a>"

    return result


def get_bezier_curve(src, dst, ctrl):

    result = f"Q {ctrl[0]} {ctrl[1]} "
    result += f"{dst[0]} {dst[1]} "
    return result
