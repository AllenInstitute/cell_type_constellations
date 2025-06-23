import matplotlib.figure
import numpy as np
import pathlib
import PIL.ImageColor

import cell_type_constellations.serialization.ingestion as ingestion_utils
import cell_type_constellations.serialization.serialization as serialization_utils
import cell_type_constellations.plotting.plotting_api as plotting_api


def main():
    version = '20250507'
    dst_h5_path = pathlib.Path(f'../../app_data/bg_aligned.for_download.{version}.h5')
    cell_to_cluster_path=f'data/{version}/cell_to_cluster_membership.csv'
    cluster_membership_path=f'data/{version}/cluster_to_cluster_annotation_membership.csv'
    cluster_annotation_path=f'data/{version}/cluster_annotation_term.csv'
    embedding_path=f'data/{version}/cell_2d_embedding_coordinates.csv'
    hierarchy=['Neighborhood', 'Class', 'Subclass', 'Group', 'Cluster']

    serialization_utils.serialize_from_csv(
            cell_to_cluster_path=cell_to_cluster_path,
            cluster_membership_path=cluster_membership_path,
            cluster_annotation_path=cluster_annotation_path,
            embedding_path=embedding_path,
            hierarchy=hierarchy,
            dst_path=dst_h5_path,
            clobber=True
        )

if __name__ == "__main__":
    main()
