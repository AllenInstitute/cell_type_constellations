import argparse
import pathlib

import cell_type_constellations.serialization.serialization as serialization_utils
import abc_atlas_access.abc_atlas_cache.abc_project_cache as cache_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dst_path', type=str, default=None
    )
    parser.add_argument(
       '--clobber', default=False, action='store_true'
    )
    args = parser.parse_args()

    if args.dst_path is None:
        raise ValueError(
            "Must specify dst_path"
        )

    abc_dir = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/"
        "lydian/ABC_handoff"
    )

    assert abc_dir.is_dir()

    abc_cache = cache_module.AbcProjectCache.from_local_cache(
        abc_dir
    )

    taxonomy_dir = 'HMBA-BG-taxonomy-CCN20250428'
    aligned_dir = 'HMBA-10xMultiome-BG-Aligned'

    cell_to_cluster_path = abc_cache.get_metadata_path(
        directory=taxonomy_dir,
        file_name='cell_to_cluster_membership'
    )

    cluster_membership_path = abc_cache.get_metadata_path(
       directory=taxonomy_dir,
       file_name='cluster_to_cluster_annotation_membership'
    )

    cluster_annotation_path = abc_cache.get_metadata_path(
        directory=taxonomy_dir,
        file_name='cluster_annotation_term'
    )

    embedding_path = abc_cache.get_metadata_path(
        directory=taxonomy_dir,
        file_name='cell_2d_embedding_coordinates'
    )

    hierarchy=['Neighborhood', 'Class', 'Subclass', 'Group', 'Cluster']

    serialization_utils.serialize_from_csv(
            cell_to_cluster_path=cell_to_cluster_path,
            cluster_membership_path=cluster_membership_path,
            cluster_annotation_path=cluster_annotation_path,
            embedding_path=embedding_path,
            hierarchy=hierarchy,
            dst_path=args.dst_path,
            clobber=args.clobber
        )

if __name__ == "__main__":
    main()
