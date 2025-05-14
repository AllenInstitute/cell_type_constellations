import numpy as np


def html_front_matter_n_columns(
        n_columns=2):
    """
    Parameters
    ----------
    n_columns:
        an integer. The number of columns to include
    """
    width_pct = np.floor(100/n_columns).astype(int)
    html = """<html>"""
    html += """<head>"""
    html += """
    <style>
    {
        box-sizing: border-box;
    }
    .column {
    """
    html += f"""
        float: left;
        width: {width_pct}%;
    """
    html += """
    }
    .row:after {
        content: "";
        display: table;
        clear: both;
    }
    </style>"""
    html += """</head>"""
    html += """<body>"""

    return html


def html_end_matter():
    html = """<p>Proof of Concept only.
            Data presented here is subject to change.
            </p>"""
    html += """</body>"""
    html += """</html>"""
    return html


def html_start_ulist():
    return """<ul>"""


def html_end_ulist():
    return """</ul>"""


def html_start_litem():
    return """<li>"""


def html_end_litem():
    return """</li>"""


def return_to_root():

    html = "<p>"
    html += go_to_constellation_landing_page()
    html += "</p>"
    html += "</p>"

    return html


def go_to_constellation_landing_page():
    html = """<a href="/constellation_plot_landing_page">"""
    html += """Go to constellation plot landing page</a></br>"""
    return html


def end_of_page():
    html = "<p>"
    html += "======================</br>"
    html += "======================</br>"
    html += return_to_root()
    html += html_end_matter()
    html += "</p>"
    return html


def overlay_style():
    """
    Copied from
    https://stackoverflow.com/questions/43806515/position-svg-elements-over-an-image
    """
    html = """
    <style>
        .img-overlay-wrap {
          position: relative;
          display: inline-block; /* <= shrinks container to image size */
        }

        .img-overlay-wrap img { /* <= optional, for responsiveness */
           display: block;
        }

        .img-overlay-wrap svg {
          position: absolute;
          top: 0;
          left: 0;
        }

    </style>
    """
    return html
