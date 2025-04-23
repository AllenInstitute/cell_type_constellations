"""
This module will define the class FieldOfView whose job it will
be to transform between 2D embedding coordinates and pixel
coordinates
"""

import h5py
import numpy as np

import cell_type_constellations.utils.coord_utils as coord_utils


class FieldOfView(object):

    def __init__(
            self,
            embedding_to_pixel,
            fov_height,
            fov_width,
            max_radius,
            min_radius):
        """
        Parameters
        ----------
        embedding_to_pixel:
            3x3 array used to transform
            [embedding_x, embedding_y, 1]
            to
            [pixel_x, pixel_y, 1]
        fov_height:
            The height (in pixels) of the field of view
        fov_width:
            The width (in pixels) of the field of view
        max_radius:
            The maximum radius (in pixel coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in pixel coordinates)
            of a node in the constellation plot
        """
        self._fov_height = fov_height
        self._fov_width = fov_width

        self._max_radius = max_radius
        self._min_radius = min_radius

        self._embedding_to_pixel = embedding_to_pixel
        self._pixel_to_embedding = np.linalg.inv(embedding_to_pixel)

    def to_hdf5(self, hdf5_path, group_path):
        """
        Write the parameters necessary to instantiate
        this FieldOfView to a specified group in an HDF5
        file
        """
        with h5py.File(hdf5_path, 'a') as dst:
            if group_path in dst:
                raise RuntimeError(
                    f"{group_path} already exists in {hdf5_path}; "
                    "unclear how to proceed"
                )
            dst_grp = dst.create_group(group_path)
            dst_grp.create_dataset(
                "embedding_to_pixel",
                data=self.embedding_to_pixel
            )
            dst_grp.create_dataset(
                "radius_bounds",
                data=np.array([self.min_radius, self.max_radius])
            )
            dst_grp.create_dataset(
                "dimensions",
                data=np.array([self.width, self.height])
            )

    @classmethod
    def from_hdf5(
            cls,
            hdf5_path,
            group_path):
        """
        Read a previously serialized FieldOfView from a group
        in an HDF5 file.

        Parameters
        ----------
        hdf5_path:
            path to the HDF5 path from which to read this FieldOfView
        group_path:
            specification of group in HDF5 file from which to read this
            FieldOfView
        """
        with h5py.File(hdf5_path, 'r') as src:
            result = cls.from_hdf5_handle(
                hdf5_handle=src,
                group_path=group_path)
        return result

    @classmethod
    def from_hdf5_handle(
            cls,
            hdf5_handle,
            group_path):

        src_grp = hdf5_handle[group_path]
        embedding_to_pixel = src_grp['embedding_to_pixel'][()]
        radius_bounds = src_grp['radius_bounds'][()]
        dimensions = src_grp['dimensions'][()]

        return cls(
            embedding_to_pixel=embedding_to_pixel,
            min_radius=radius_bounds[0],
            max_radius=radius_bounds[1],
            fov_width=dimensions[0],
            fov_height=dimensions[1]
        )

    @classmethod
    def from_h5ad(
            cls,
            h5ad_path,
            coord_key,
            fov_height,
            max_radius,
            min_radius):
        """
        Instantiate a FieldOfView from data stored in a single
        h5ad file.

        Parameters
        ----------
        h5ad_path:
            Path to the h5ad file
        coord_key:
            key under obsm where the embedding coordinates are
        fov_height:
            The height (in pixels) of the field of view
        max_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        """
        coords = coord_utils.get_coords_from_h5ad(
            h5ad_path=h5ad_path,
            coord_key=coord_key
        )
        return cls.from_coords(
            coords=coords,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius
        )

    @classmethod
    def from_coords(
            cls,
            coords,
            fov_height,
            max_radius,
            min_radius):
        """
        Instantiate a FieldOfView from a numpy array
        of visualization embedding coordinates

        Parameters
        ----------
        coords:
            The (n_cells, 2) np.ndarray of embedding coordinates
            in which we are going to visualize the constellation
            plot
        fov_height:
            The height (in pixels) of the field of view
        max_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        """
        if coords.shape[1] != 2:
            raise RuntimeError(
                "Embedding coordinates for FieldOfView must "
                f"be 2-dimensional; you gave {coords.shape}"
            )

        bounds = np.array([
            [coords[:, 0].min(), coords[:, 0].max()],
            [coords[:, 1].min(), coords[:, 1].max()]
        ])
        return cls.from_embedding_bounds(
            embedding_bounds=bounds,
            fov_height=fov_height,
            max_radius=max_radius,
            min_radius=min_radius
        )

    @classmethod
    def from_embedding_bounds(
           cls,
           embedding_bounds,
           fov_height,
           max_radius,
           min_radius):
        """
        Instantiate a FieldOfView from a numpy array
        containing the min/max values of a set of visualization
        embedding coordinates.

        Parameters
        ----------
        embedding_bounds:
            2x2 numpy array that is
            [[xmin, xmax],
             [ymin, ymax]]
            of embedding space coordinates in this field of view
        fov_height:
            The height (in pixels) of the field of view
        max_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        min_radius:
            The maximum radius (in embedding coordinates)
            of a node in the constellation plot
        """

        xmax = embedding_bounds[0, 1]
        xmin = embedding_bounds[0, 0]

        ymax = embedding_bounds[1, 1]
        ymin = embedding_bounds[1, 0]
        dx = 6*max_radius + xmax - xmin
        dy = 6*max_radius + ymax - ymin

        ratio = dx/dy
        fov_width = np.round(fov_height*ratio).astype(int)

        pixel_buffer = 3*max_radius//2
        pixel_origin = np.array([pixel_buffer, pixel_buffer])
        pixel_extent = np.array([fov_width-2*pixel_buffer,
                                 fov_height-2*pixel_buffer])

        # origin and extent in embedding coordinates
        embedding_origin = [xmin, ymin]
        embedding_extent = [xmax-xmin, ymax-ymin]

        embedding_to_pixel = get_embedding_to_pixel(
            pixel_origin=pixel_origin,
            pixel_extent=pixel_extent,
            embedding_origin=embedding_origin,
            embedding_extent=embedding_extent
        )

        return cls(
            fov_height=fov_height,
            fov_width=fov_width,
            max_radius=max_radius,
            min_radius=min_radius,
            embedding_to_pixel=embedding_to_pixel
        )

    @property
    def width(self):
        """
        height of field of view in pixel coordinates
        """
        return self._fov_width

    @property
    def height(self):
        """
        height of field of view in pixel coordinates
        """
        return self._fov_height

    @property
    def max_radius(self):
        """
        max radius in pixel coordinates of a node in the
        constellation plot
        """
        return self._max_radius

    @property
    def min_radius(self):
        """
        min radius in pixel coordinates of a node in the
        constellation plot
        """
        return self._min_radius

    @property
    def embedding_to_pixel(self):
        """
        The 3x3 transformation matrix that can transform
        an [x, y, 1] vector in embedding coordinates to
        an [px, py, 1] vector in pixel coordinates
        """
        return self._embedding_to_pixel

    @property
    def pixel_to_embedding(self):
        """
        The 3x3 transformation matrix that can transform
        an [px, py, 1] vector in pixel coordinates to
        an [x, y, 1] vector in embedding coordinates
        """
        return self._pixel_to_embedding

    def transform_to_pixel_coordinates(
            self,
            embedding_coords):
        """
        Transform an array of embedding coordinates into
        pixel coordinates.

        Parameters
        ----------
        embedding_coords:
            an (N, 2) array of embedding coordinates
            to be transformed

        Returns
        -------
        pixel_coords:
            an (N, 2) array that is the input embedding
            coordinates transformed into pixel coordinates
        """
        return apply_transformation(
            input_coords=embedding_coords,
            transformation_matrix=self.embedding_to_pixel
        )

    def transform_to_embedding_coordinates(
            self,
            pixel_coords):
        """
        Transform an array of pixel coordinates into
        embedding coordinates.

        Parameters
        ----------
        pixel_coords:
            an (N, 2) array of pixel coordinates
            to be transformed

        Returns
        -------
        embedding_coords:
            an (N, 2) array that is the input pixel
            coordinates transformed into embedding coordinates
        """
        return apply_transformation(
            input_coords=pixel_coords,
            transformation_matrix=self.pixel_to_embedding
        )

    def get_pixel_radii(self, n_cells_array, n_cells_max):
        """
        Take an array of n_cells values and return an array
        of correspondingly scaled node radii (in pixel coords)

        Parameters
        ----------
        n_cells_array:
            The array of n_cells values for which to calculate
            pixel radii
        n_cell_max:
            The maximum theoretical value of n_cells in this
            constellation plot (for consistent scaling)

        Returns
        -------
        pixel_radii:
            an array of pixel space radii corresponding to the
            values in n_cells_array
        """
        dr = self.max_radius-self.min_radius
        logarithmic_r = np.log2(1.0+n_cells_array/n_cells_max)
        pixel_radii = self.min_radius+dr*logarithmic_r
        return pixel_radii


def get_embedding_to_pixel(
        pixel_origin,
        pixel_extent,
        embedding_origin,
        embedding_extent):
    """
    Set the transformation matrix for converting from
    embedding coordinates to pixel coordinates.
    """

    e0 = pixel_extent[0]
    e1 = pixel_extent[1]
    we0 = embedding_extent[0]
    we1 = embedding_extent[1]
    p0 = pixel_origin[0]
    p1 = pixel_origin[1]
    w0 = embedding_origin[0]
    w1 = embedding_origin[1]

    embedding_to_pixel = np.array([
        [e0/we0, 0.0, p0-e0*w0/we0],
        [0.0, -e1/we1, p1+e1*(w1+we1)/we1],
        [0.0, 0.0, 1.0]
    ])

    return embedding_to_pixel


def apply_transformation(
        input_coords,
        transformation_matrix):
    """
    Apply transformation, either from pixel to
    embedding coords or vice versa

    Parameters
    ----------
    input_coords:
        an (N, 2) array of embedding coordinates
        to be transformed
    transformation_matrix:
        (3, 3) transformation matrix to be applied to the
        coordinates

    Returns
    -------
    output_coords:
        an (N, 2) array that is the transformed coordinates
    """
    if len(input_coords.shape) != 2 or input_coords.shape[1] != 2:
        raise RuntimeError(
           "input_coords must be an (N, 2) array; "
           f"yours has shape {input_coords.shape}"
        )
    input3d = np.vstack([
        input_coords.transpose(),
        np.ones(input_coords.shape[0])])
    raw = np.dot(transformation_matrix, input3d).transpose()
    return raw[:, :-1]
