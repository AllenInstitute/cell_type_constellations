import argparse
import cherrypy
import pathlib

import cell_type_constellations
import cell_type_constellations.rendering.rendering_api as rendering_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="Ip of host to serve the app on."
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help="Port to serve the app on."
    )
    args = parser.parse_args()

    cherrypy.server.socket_host = args.host
    cherrypy.server.socket_port = args.port

    cherrypy.quickstart(
        Visualizer()
    )


class Visualizer(object):

    def __init__(self):

        file_path = pathlib.Path(
            cell_type_constellations.__file__)

        self.data_dir = file_path.parent.parent.parent / 'app_data'

        if not self.data_dir.is_dir():
            raise RuntimeError(
                f"Data dir {self.data_dir} is not a dir"
            )

        self.constellation_plot_config = (
            rendering_api.get_constellation_plot_config(
                self.data_dir
            )
        )

    @cherrypy.expose
    def index(self):
        return self.constellation_plot_landing_page()

    @cherrypy.expose
    def constellation_plot_landing_page(self):

        html = "<p>Choose taxonomy for constellation plot</p>"
        taxonomy_name_list = list(self.constellation_plot_config.keys())
        taxonomy_name_list.sort()
        for taxonomy_name in taxonomy_name_list:
            html += f"""<a href="/constellation_plot?taxonomy_name={taxonomy_name}&default=true">"""  # noqa: E501
            html += f"""{taxonomy_name}</a></br>"""
        return html

    @cherrypy.expose
    def constellation_plot(
            self,
            taxonomy_name=None,
            centroid_level=None,
            color_by=None,
            scatter_plot_level=None,
            connection_coords=None,
            default=False,
            show_centroid_labels='true'):

        if show_centroid_labels == 'true':
            show_centroid_labels = True
        else:
            show_centroid_labels = False

        config = self.constellation_plot_config[taxonomy_name]

        hdf5_path = pathlib.Path(config['path'])

        if default:
            centroid_level = config['centroid_level']
            color_by = config['color_by']
            scatter_plot_level = config['scatter_plot_level']
            connection_coords = config['connection_coords']

        html = rendering_api.constellation_svg_from_hdf5(
                hdf5_path=hdf5_path,
                centroid_level=centroid_level,
                scatter_plot_level=scatter_plot_level,
                color_by=color_by,
                connection_coords=connection_coords,
                show_centroid_labels=show_centroid_labels)

        return html


if __name__ == "__main__":
    main()
