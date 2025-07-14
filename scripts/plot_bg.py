import matplotlib.figure
import pathlib

import cell_type_constellations.plotting.plotting_api as plotting_api


def main():
    hdf5_path = pathlib.Path(
        "../app_data/bg.aligned.official.colors.20250623.h5"
    )
    assert hdf5_path.is_file()

    fig = matplotlib.figure.Figure(figsize=(40, 40))
    axis = fig.add_subplot(1,1,1)
    plotting_api.plot_constellation_in_mpl(
        hdf5_path=hdf5_path,
        centroid_level='Cluster',
        color_by_level='Group',
        connection_coords='embedding',
        hull_level=None,
        axis=axis,
        dst_path=None,
        fill_hulls=False,
        scatter_plot_level=0.9,
        centroid_factor=0.5
    )
    axis.axis('off')
    fig.tight_layout()
    fig.savefig(
        'figures/bg.constellation.color.png'
    )


if __name__ == "__main__":
    main()
