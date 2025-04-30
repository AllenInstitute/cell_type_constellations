"""
Utilities to create CellSet and embedding coordinates
from a data release set of csvs
"""
import numpy as np
import pandas as pd

import cell_type_constellations.cells.cell_set as cell_set_utils


def from_csv(
        cell_to_cluster_path,
        cluster_annotation_path,
        cluster_membership_path,
        embedding_path,
        hierarchy):


    alias_to_membership = _get_alias_to_membership(
        cluster_membership_path
    )

    color_lookup = _get_color_lookup(
        cluster_annotation_path
    )

    coord_lookup = _get_coord_lookup(
        embedding_path
    )

    cell_to_alias = _get_cell_to_alias(
        cell_to_cluster_path
    )

    embedding_coords = []
    cell_metadata = []

    cell_label_list = sorted(cell_to_alias.keys())
    for cell in cell_label_list:
        alias = cell_to_alias[cell]
        if alias not in alias_to_membership:
            continue
        membership = alias_to_membership[alias]
        embedding_coords.append(
            coord_lookup[cell]
        )
        metadata = {'cell_label': cell}
        metadata.update(membership)
        cell_metadata.append(metadata)

    cell_metadata = pd.DataFrame(cell_metadata)
    cell_set = cell_set_utils.CellSet(
        cell_metadata=cell_metadata,
        discrete_fields=hierarchy,
        continuous_fields=None,
        leaf_field=hierarchy[-1]
    )

    return {
        'cell_set': cell_set,
        'embedding_coords': np.array(embedding_coords),
        'discrete_color_lookup': color_lookup
    }


def _get_alias_to_membership(cluster_membership_path):

    cluster_membership = pd.read_csv(
        cluster_membership_path
    )[['cluster_annotation_term_name',
       'cluster_annotation_term_label',
       'cluster_annotation_term_set_name',
       'cluster_annotation_term_set_label',
       'cluster_alias']].to_dict(orient='records')

    alias_to_membership = dict()
    for row in cluster_membership:
        alias = row['cluster_alias']
        if alias not in alias_to_membership:
            alias_to_membership[alias] = dict()
        set_name = row['cluster_annotation_term_set_name']
        set_label = row['cluster_annotation_term_set_label']
        name = row['cluster_annotation_term_name']
        label = row['cluster_annotation_term_label']
        assert set_name not in alias_to_membership[alias]
        assert set_label not in alias_to_membership[alias]
        alias_to_membership[alias][set_name] = name
        alias_to_membership[alias][set_label] = label

    return alias_to_membership


def _get_color_lookup(cluster_annotation_path):
    df = pd.read_csv(cluster_annotation_path).to_dict(orient='records')
    color_lookup = dict()
    for row in df:
        set_label = row['cluster_annotation_term_set_label']
        set_name = row['cluster_annotation_term_set_name']
        label = row['label']
        name = row['name']
        color = row['color_hex_triplet']
        if set_label not in color_lookup:
            color_lookup[set_label] = dict()
        if set_name not in color_lookup:
            color_lookup[set_name] = dict()

        color_lookup[set_name][name] = color
        color_lookup[set_label][label] = color

    return color_lookup


def _get_coord_lookup(embedding_path):
    df = pd.read_csv(embedding_path)
    coord_arr = df[['x', 'y']].to_numpy()
    cell_label_list = df.cell_label.values
    result = {
        cell_label: coord_arr[ii, :]
        for ii, cell_label in enumerate(cell_label_list)
    }
    return result


def _get_cell_to_alias(cell_to_cluster_path):
    df = pd.read_csv(cell_to_cluster_path)
    result = {
        cell_label: cluster_alias
        for cell_label, cluster_alias
        in zip(df.cell_label.values, df.cluster_alias.values)
    }
    return result
