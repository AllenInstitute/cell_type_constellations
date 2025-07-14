import json
import pathlib

import cell_type_constellations.serialization.serialization as serialization

def main():

    h5ad_path = pathlib.Path(
        "/allen/programs/celltypes/workgroups/rnaseqanalysis/"
        "EvoGen/SpinalCord/manuscript/RNA/"
        "AIBS_SpC_consensus_taxonomy_harmonized_AIT-pre-print.h5ad"
    )

    assert h5ad_path.is_file()

    with open('spinal_cord_colors.json', 'rb') as src:
        colors = json.load(src)


    dst_path = pathlib.Path(
        "../app_data/spinal_cord.20250714.h5"
    )

    coords = 'X_umap_species_integrated_final'
    hierarchy = [
        'consensus_cluster',
        'Group',
        'Subclass',
        'Class'
    ]

    serialization.serialize_from_h5ad(
        h5ad_path=h5ad_path,
        visualization_coords=coords,
        connection_coords_list=[coords],
        discrete_fields=hierarchy,
        discrete_color_map=colors,
        continuous_fields=None,
        leaf_field=hierarchy[0],
        dst_path=dst_path,
        tmp_dir='/local1/scott_daniel/scratch',
        k_nn=15,
        n_processors=4,
        fov_height=1080,
        max_radius=35,
        min_radius=5,
        clobber=True
    )
    print(f'wrote {dst_path}')


if __name__ == "__main__":
    main()
