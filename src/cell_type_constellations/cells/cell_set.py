"""
This module will define the class CellSet, which associates cells
with arrays of metadata fields/annotations for purposes of determining
different node and edge arrangements in the constellation plot
"""

import copy
import numpy as np

import cell_type_mapper.utils.anndata_utils as anndata_utils
import cell_type_constellations.cells.tree_utils as tree_utils


class CellSet(object):

    def __init__(
            self,
            cell_metadata,
            discrete_fields,
            continuous_fields,
            leaf_field=None):
        """
        Parameters
        ----------
        cell_metadata:
            a pandas DataFrame of the metadata and annotations
            associated with each cell (obs of an h5ad file)
        discrete_fields:
            a list of columns in cell_metadata by which the cells
            are to be discretely clustered (i.e. the "taxonomic types")
        continuous_fields:
            a list of columns in cell_metadata that are to be treated
            as numerical value whose statistics are to be grouped
            along the discrete fields
        leaf_field:
            the (optional) discrete_field to be interpreted as the
            "leaf level" of the taxonomy
        """

        if discrete_fields is None:
            discrete_fields = []
        if continuous_fields is None:
            continuous_fields = []

        # infer child-to-parent relationships, i.e. relationships
        # between discrete_fields where the value in one (the child)
        # necessarily implies the value in another (the parent)
        self._child_to_parent = tree_utils.infer_tree(
            cell_metadata=cell_metadata,
            discrete_fields=discrete_fields
        )

        self._type_field_list = copy.deepcopy(discrete_fields)
        self._continuous_field_list = copy.deepcopy(continuous_fields)

        if leaf_field is not None:
            if leaf_field not in self._type_field_list:
                raise RuntimeError(
                    f"Leaf field {leaf_field} is not in your "
                    "list of discrete_fields"
                )
        self._leaf_type = leaf_field

        self._n_cells = len(cell_metadata)

        # map values in discrete_fields to the indexes of cells
        # that belong to those values
        self._type_masks = dict()
        self._statistics = dict()
        self._idx_to_types = dict()
        self._n_cells_lookup = dict()
        for col in discrete_fields:
            self._type_masks[col] = dict()
            self._statistics[col] = dict()
            self._idx_to_types[col] = cell_metadata[col].values
            self._n_cells_lookup[col] = dict()

            unq_value_list = np.unique(cell_metadata[col].values)
            for unq in unq_value_list:
                idx = np.where(cell_metadata[col].values == unq)[0]
                self._n_cells_lookup[col][unq] = len(idx)
                if len(idx) >= 2:
                    ddof = 1
                else:
                    ddof = 0
                self._type_masks[col][unq] = idx
                self._statistics[col][unq] = dict()
                for stat_col in continuous_fields:
                    stats = {
                        'mean': np.mean(
                            cell_metadata[stat_col].values[idx]),
                        'var': np.var(
                            cell_metadata[stat_col].values[idx],
                            ddof=ddof)
                    }
                    self._statistics[col][unq][stat_col] = stats

        if self.leaf_type is not None:
            self._create_parent_to_leaves()

    @classmethod
    def from_h5ad(
            cls,
            h5ad_path,
            discrete_fields,
            continuous_fields,
            leaf_field=None):
        """
        Instantiate a CellSet from an h5ad file

        Parameters
        ----------
        h5ad_path:
            path to the h5ad file
        discrete_fields:
            a list of columns in cell_metadata by which the cells
            are to be discretely clustered (i.e. the "taxonomic types")
        continuous_fields:
            a list of columns in cell_metadata that are to be treated
            as numerical value whose statistics are to be grouped
            along the discrete fields
        leaf_field:
            the (optional) discrete_field to be interpreted as the
            "leaf level" of the taxonomy
        """
        cell_metadata = anndata_utils.read_df_from_h5ad(
            h5ad_path,
            df_name='obs'
        )

        # make sure all discrete fields are recorded as strings
        for col in discrete_fields:
            cell_metadata[col] = cell_metadata[col].astype(str)

        return cls(
            cell_metadata=cell_metadata,
            discrete_fields=discrete_fields,
            continuous_fields=continuous_fields,
            leaf_field=leaf_field)

    @property
    def n_cells(self):
        """
        Total number of cells in this CellSet
        """
        return self._n_cells

    @property
    def leaf_type(self):
        """
        The type_field that is to be interpreted as the
        leaf of the taxonomy
        """
        return self._leaf_type

    def n_cells_in_type(self, type_field, type_value):
        """
        Number of cells assigned to the (type_field, type_value)
        pair
        """
        if type_field not in self._type_masks:
            raise RuntimeError(
                f"No cell types associated with field {type_field}"
            )
        lookup = self._n_cells_lookup[type_field]
        if type_value not in lookup:
            raise RuntimeError(
                f"{type_value} not a valid value for field {type_field}"
            )
        return lookup[type_value]

    def type_field_list(self):
        """
        List of valid type fields
        """
        return self._type_field_list

    def continuous_field_list(self):
        """
        List of valid continuous fields
        """
        return self._continuous_field_list

    def type_value_list(self, type_field):
        """
        Return the list of 'types' derived from a discrete
        field in the cell metadata
        """
        if type_field not in self._type_masks:
            raise RuntimeError(
                f"No cell types associated with field {type_field}"
            )
        result = sorted(self._type_masks[type_field])
        return result

    def type_mask(self, type_field, type_value):
        """
        Return the array of cell indices (i.e. the rows of obs)
        associated with a specific value in a specific field
        of the cell_metadata
        """
        if type_field not in self._type_masks:
            raise RuntimeError(
                f"No cell types associated with field {type_field}"
            )
        lookup = self._type_masks[type_field]
        if type_value not in lookup:
            raise RuntimeError(
                f"{type_value} not a valid value for field {type_field}"
            )
        return lookup[type_value]

    def type_value_from_idx(self, type_field, idx_array):
        """
        Given a type_field and an array of integers,
        return the type_values associated by the cells at those
        integers.
        """
        if type_field not in self._type_masks:
            raise RuntimeError(
                f"No cell types associated with field {type_field}"
            )
        return self._idx_to_types[type_field][idx_array]

    def stat_field_list(self, type_field, type_value):
        """
        Return the list of valid stat fields for a given
        type_filed, type_value pair
        """
        if type_field not in self._statistics:
            raise RuntimeError(
                f"No cell types associated with field {type_field}"
            )
        lookup = self._statistics[type_field]
        if type_value not in lookup:
            raise RuntimeError(
                f"{type_value} not a valid value for field {type_field}"
            )
        return sorted(lookup[type_value].keys())

    def stats(self, type_field, type_value, stat_field):
        """
        Return the stats dict for a specific set of
            (type_field,
             type_value,
             stat_field)
        """
        if type_field not in self._statistics:
            raise RuntimeError(
                f"No cell types associated with field {type_field}"
            )
        lookup = self._statistics[type_field]
        if type_value not in lookup:
            raise RuntimeError(
                f"{type_value} not a valid value for field {type_field}"
            )
        if stat_field not in lookup[type_value]:
            raise RuntimeError(
                f"{stat_field} not a valid statistics field"
            )
        return lookup[type_value][stat_field]

    def parent_annotations(self, type_field, type_value):
        """
        Return a dict mapping type_field: type_value for any
        "parents" of the specified (type_field, type_value) pair
        (where by "parents" we mean "other types that are precisely
        implied by the specified type).

        This dict will includ the type_field:type_value mapping
        for self (the specified taxon) so that we can consistently
        encode labels for the centroids.
        """
        result = {type_field: type_value}
        if type_field not in self._child_to_parent:
            return result
        for parent_field in self._child_to_parent[type_field]:
            parent_value = (
                self._child_to_parent[type_field][parent_field][type_value]
            )
            result[parent_field] = parent_value
        return result

    def _create_parent_to_leaves(self):
        # just need dict that takes a parent_field, parent_value and gives
        # a list of leaves
        self._parent_to_leaves = dict()
        for leaf_value in self.type_value_list(self.leaf_type):
            parentage = self.parent_annotations(
                type_field=self.leaf_type,
                type_value=leaf_value)
            for parent_field in parentage:
                if parent_field not in self._parent_to_leaves:
                    self._parent_to_leaves[parent_field] = dict()
                parent_value = parentage[parent_field]
                if parent_value not in self._parent_to_leaves[parent_field]:
                    self._parent_to_leaves[parent_field][parent_value] = []
                self._parent_to_leaves[parent_field][parent_value].append(
                    leaf_value
                )

        # validate that, if a type_field is in parent_to_leaves, all of
        # its values are also present
        for type_field in self.type_field_list():
            if type_field not in self._parent_to_leaves:
                continue
            for type_value in self.type_value_list(type_field):
                assert type_value in self._parent_to_leaves[type_field]

    def parent_to_leaves(self, type_field, type_value):
        if self.leaf_type is None:
            raise RuntimeError(
                "You have not specified a leaf_type for this CellSet"
            )
        if type_field not in self._parent_to_leaves:
            return []
        if type_value not in self._parent_to_leaves[type_field]:
            return []
        return copy.deepcopy(self._parent_to_leaves[type_field][type_value])
