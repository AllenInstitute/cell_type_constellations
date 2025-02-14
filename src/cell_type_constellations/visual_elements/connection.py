"""
This module defines the Connection class which represents a connection
between to nodes in the constellation plot.

Note: Connections are instantiated with PixelSpaceCentroids. Because
of how they are rendered, you need to know how the centroids will appear
in actual pixel space, not in embedding space
"""

import h5py
import numpy as np

import cell_type_constellations.utils.geometry_utils as geometry_utils
import cell_type_constellations.utils.connection_utils as connection_utils

from cell_type_constellations.visual_elements.centroid import (
    PixelSpaceCentroid
)

def get_connection_list(
        pixel_centroid_lookup,
        mixture_matrix_file_path,
        type_field):
    """
    Get a list of rendering-ready connections for a
    visualization

    Parameters
    ----------
    pixel_centroid_lookup:
        a dict mapping type_field, type_value pairs to
        PixelSpaceCentroids
    mixture_matrix_file_path:
        path to the HDF5 file containing the mixture matrices
        associating centroids in this visualization
    type_field:
        the type_field (e.g class, subclass, supertype) for which
        to get the connections

    Returns
    -------
    A list of Connections that are ready to be rendered
    """

    with h5py.File(mixture_matrix_file_path, 'r') as src:
        mixture_matrix = src[type_field]['mixture_matrix'][()]
        type_value_list = [
            val.decode('utf-8')
            for val in src[type_field]['row_key'][()]
        ]
        k_nn = src['k_nn'][()]

    centroid_list = [
        pixel_centroid_lookup[type_field][val]
        for val in type_value_list
    ]

    n_cells_array = np.array(
        [centroid.n_cells for centroid in centroid_list]
    )

    valid_connections = connection_utils.choose_connections(
        mixture_matrix=mixture_matrix,
        n_cells=n_cells_array,
        k_nn=k_nn
    )

    # make sure each pair is unique, regardless of order
    loaded_connections = set()
    connection_list = []
    max_ratio = None
    for i0, i1 in zip(*valid_connections):

        pair = tuple(sorted((i0, i1)))

        if pair in loaded_connections:
            continue
        loaded_connections.add(pair)

        n0 = mixture_matrix[i0, i1]/centroid_list[i0].n_cells
        n1 = mixture_matrix[i1, i0]/centroid_list[i1].n_cells

        this_max = max(n0, n1)
        if max_ratio is None or this_max > max_ratio:
            max_ratio = this_max

        if n0 > n1:
            i_src = i0
            i_dst = i1
            n_src = mixture_matrix[i0, i1]
            n_dst = mixture_matrix[i1, i0]
        else:
            i_src = i1
            i_dst = i0
            n_src = mixture_matrix[i1, i0]
            n_dst = mixture_matrix[i0, i1]

        src = centroid_list[i_src]
        dst = centroid_list[i_dst]
        conn = Connection(
            src_centroid=src,
            dst_centroid=dst,
            n_src_neighbors=n_src,
            n_dst_neighbors=n_dst,
            k_nn=k_nn
        )
        connection_list.append(conn)

    for conn in connection_list:
        conn.set_rendering_corners(
            max_connection_ratio=max_ratio
        )

    bezier_control_points = get_bezier_control_points(
        connection_list=connection_list)

    for ii, bez in enumerate(bezier_control_points):
        connection_list[ii].set_bezier_control_points(bez)

    return connection_list


class Connection(object):

    def __init__(
            self,
            src_centroid,
            dst_centroid,
            n_src_neighbors,
            n_dst_neighbors,
            k_nn):
        """
        Parameters
        ----------
        src_centroid:
            a PixelSpaceCentroid representing the source of the
            Connection
        dst_centroid:
            a PixelSpaceCentroid representing the destination of
            the Connection
        n_src_neighbors:
            an int. How many of src_centroid's nearest neighbors
            point to dst_centroid
        n_dst_neighbors:
            an int. How many of dst_centroid's nearest neighbors
            point to src_centroid
        k_nn:
            an int. How many nearest neighbors of each cell did
            you query for when creating the mixture matrix used
            to derive this Connection.
        """
        if not isinstance(src_centroid, PixelSpaceCentroid):
            raise RuntimeError(
                "src_centroid must be a PixelSpaceCentroid, not "
                f"{type(src_centroid)}"
            )
        if not isinstance(dst_centroid, PixelSpaceCentroid):
            raise RuntimeError(
                "dst_centroid must be a PixelSpaceCentroid, not "
                f"{type(dst_centroid)}"
            )

        self._src = src_centroid
        self._dst = dst_centroid
        self._n_src_neighbors = n_src_neighbors
        self._n_dst_neighbors = n_dst_neighbors
        self._rendering_corners = None
        self._bezier_control_points = None
        self._k_nn = k_nn

    @property
    def ready_to_render(self):
        """
        boolean indicating if this centroid is ready to be rendered
        """
        return (
            (self._rendering_corners is not None)
            and (self._bezier_control_points is not None)
        )

    @property
    def src(self):
        """
        rhe PixelSpaceCentroid of the Connection's
        source node
        """
        return self._src

    @property
    def dst(self):
        """
        the PixelSpaceCentroid of the Connection's
        destination node
        """
        return self._dst

    @property
    def k_nn(self):
        """
        the number of nearest neighbors per cell used to
        generate the mixture matrix used in inferring these
        connecitons
        """
        return self._k_nn

    @property
    def n_src_neighbors(self):
        """
        the number of src's nearest neighbors that mapped
        to dst
        """
        return self._n_src_neighbors

    @property
    def n_dst_neighbors(self):
        """
        the number of dst's nearest neighbors that mapped to
        src
        """
        return self._n_dst_neighbors

    @property
    def src_neighbor_fraction(self):
        """
        the fraction of total possible nearest neighbors for src
        that mapped to dst
        """
        return self.n_src_neighbors/(self.src.n_cells*self.k_nn)

    @property
    def dst_neighbor_fraction(self):
        """
        the fraction of total possible nearest neighbors for dst
        that mapped to src
        """
        return self.n_dst_neighbors/(self.dst.n_cells*self.k_nn)

    @property
    def rendering_corners(self):
        """
        List of pixel space coordinates of the corners of
        the connection (if the Connection were a rectangle)
        """
        return self._rendering_corners

    @property
    def bezier_control_points(self):
        """
        The control points for the Bezier curves
        of the two curved edges of the Connection
        """
        return self._bezier_control_points

    @property
    def src_mid(self):
        """
        mid point of connection's intersection with circumference
        of src circle (relative to src center)
        """
        if not hasattr(self, '_src_mid'):
            self._find_mid_pt()
        return self._src_mid

    @property
    def dst_mid(self):
        """
        mid point of connection's intersection with circumference
        of src circle (relative to src center)
        """
        if not hasattr(self, '_dst_mid'):
            self._find_mid_pt()
        return self._dst_mid

    def _find_mid_pt(self):

        src_pt = self.src.center_pt
        dst_pt = self.dst.center_pt

        connection = dst_pt-src_pt

        norm = np.sqrt((connection**2).sum())

        self._src_mid = self.src.radius*connection/norm
        self._dst_mid = -self.dst.radius*connection/norm

    def set_rendering_corners(self, max_connection_ratio):
        """
        max_connection_ratio is the theoretical maximum
        of neighbors/n_cells for all connection endpoints in
        this visualization.
        """

        self._rendering_corners = _intersection_points(
            src_pt=self.src.center_pt,
            src_mid=self.src_mid,
            src_n_cells=self.src.n_cells,
            src_n_neighbors=self.n_src_neighbors,
            src_r=self.src.radius,
            dst_pt=self.dst.center_pt,
            dst_mid=self.dst_mid,
            dst_n_cells=self.dst.n_cells,
            dst_n_neighbors=self.n_dst_neighbors,
            dst_r=self.dst.radius,
            max_connection_ratio=max_connection_ratio)

        points = self._rendering_corners
        if geometry_utils.do_intersect([points[0], points[1]],
                        [points[2], points[3]]):
            print(f'huh {self.src.name} {self.dst.name}')

    def set_bezier_control_points(self, thermal_control):
        """
        Thermal control is the result of the get_bezier_control_points
        function run on all the connections in the field of view
        """
        assert self.rendering_corners is not None
        mid_pt = 0.5*(self.src.center_pt+self.dst.center_pt)
        dd = thermal_control-mid_pt
        ctrl0 = dd+0.5*(self.rendering_corners[0]+self.rendering_corners[1])
        ctrl1 = dd+0.5*(self.rendering_corners[2]+self.rendering_corners[3])
        self._bezier_control_points = np.array([ctrl0, ctrl1])


def _intersection_points(
        src_pt,
        src_mid,
        src_n_cells,
        src_n_neighbors,
        src_r,
        dst_pt,
        dst_mid,
        dst_n_cells,
        dst_n_neighbors,
        dst_r,
        max_connection_ratio):

    min_width = 0.25

    src_theta = 0.5*np.pi*(src_n_neighbors/(src_n_cells*max_connection_ratio))
    dst_theta = 0.5*np.pi*(dst_n_neighbors/(dst_n_cells*max_connection_ratio))

    if min_width < 2.0*src_r:
        actual_width = 2.0*src_r*np.abs(np.sin(src_theta))
        if actual_width < min_width:
            new_theta = np.asin(0.5*min_width/src_r)
            new_theta = np.sign(src_theta)*new_theta
            src_theta = new_theta

    if min_width < 2.0*dst_r:
        actual_width = 2.0*dst_r*np.abs(np.sin(dst_theta))
        if actual_width < min_width:
            new_theta = np.asin(0.5*min_width/dst_r)
            new_theta = np.sign(dst_theta)*new_theta
            dst_theta = new_theta

    src0 = src_pt + geometry_utils.rot(src_mid, src_theta)
    src1 = src_pt + geometry_utils.rot(src_mid, -src_theta)

    dst0 = dst_pt + geometry_utils.rot(dst_mid, -dst_theta)
    dst1 = dst_pt + geometry_utils.rot(dst_mid, dst_theta)

    if geometry_utils.do_intersect([src0, dst0], [dst1, src1]):
        points = [src0, dst1, dst0, src1]
    else:
        points = [src0, dst0, dst1, src1]

    return np.array(points)


def get_bezier_control_points(
        connection_list):
    """
    Take a list of Connections. Find the control points for the Bezier
    curved edges of the connections by modeling those control points as
    charged particles and having them repel each other so that the curved
    connections don't "get in each other's way"

    Returns a list of control points, one for each connection (the Connection
    needs to transform these into the two control points, one for the "upper"
    edge, one for the "lower" edge).
    """

    n_conn = len(connection_list)
    background = np.zeros((3*n_conn, 2), dtype=float)
    orthogonals = np.zeros((n_conn, 2), dtype=float)
    distances = np.zeros(n_conn, dtype=float)
    charges = np.zeros(3*n_conn, dtype=float)
    for i_conn, conn in enumerate(connection_list):
        background[i_conn*2, :] = conn.src.center_pt
        background[1+i_conn*2, :] = conn.dst.center_pt
        charges[i_conn*2] = 5.0
        charges[i_conn*2] = 5.0

        background[2*n_conn+i_conn, :] = 0.5*(conn.src.center_pt
                                              + conn.dst.center_pt)
        charges[2*n_conn+i_conn] = 1.0

        dd = conn.dst.center_pt-conn.src.center_pt
        distances[i_conn] = np.sqrt(
            (dd**2).sum()
        )
        dd = dd/distances[i_conn]
        orthogonals[i_conn, :] = geometry_utils.rot(dd, 0.5*np.pi)

    max_displacement = 0.05
    n_iter = 3
    mask = np.ones(3*n_conn, dtype=bool)
    n_tot = 0
    n_adj = 0
    for i_iter in range(n_iter):
        for i_conn in range(n_conn):
            mask[2*n_conn+i_conn] = False
            mask[i_conn*2] = False
            mask[1+i_conn*2] = False
            test_pt = background[2*n_conn+i_conn, :]
            force = compute_force(
                test_pt=test_pt,
                background_points=background[mask, :],
                charges=charges[mask]
            )
            ortho_force = np.dot(force, orthogonals[i_conn, :])
            force = 100.0*ortho_force*orthogonals[i_conn, :]

            displacement = np.sqrt((force**2).sum())
            if displacement > max_displacement*distances[i_conn]:
                force = max_displacement*distances[i_conn]*force/displacement
                n_adj += 1

            background[2*n_conn+i_conn, :] = test_pt + force
            mask[2*n_conn+i_conn] = True
            mask[i_conn*2] = True
            mask[1+2*i_conn] = True
            if displacement > 1.0e-3:
                n_tot += 1
        print(f'adj {n_adj} of {n_tot}')

    return background[2*n_conn:, :]


def compute_force(
        test_pt,
        background_points,
        charges,
        eps=0.001):

    vectors = test_pt-background_points
    rsq = (vectors**2).sum(axis=1)
    rsq = np.where(rsq > eps, rsq, eps)
    weights = charges/np.power(rsq, 2.0)
    force = (vectors.transpose()*weights).sum(axis=1)
    return force
