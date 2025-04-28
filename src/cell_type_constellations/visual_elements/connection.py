"""
This module defines the Connection class which represents a connection
between to nodes in the constellation plot.

Note: Connections are instantiated with PixelSpaceCentroids. Because
of how they are rendered, you need to know how the centroids will appear
in actual pixel space, not in embedding space
"""

import h5py
import numpy as np

import cell_type_constellations.plotting.bezier as bezier_utils
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

    bezier_control_points = get_bezier_control_points(
        centroid_list=centroid_list,
        connection_list=connection_list)

    for conn, bez in zip(connection_list, bezier_control_points):
        conn.thermal_control_point = bez

    for conn in connection_list:
        conn.set_rendering_corners(
            max_connection_ratio=max_ratio
        )

    for conn in connection_list:
        conn.set_bezier_control_points()

    return connection_list


def write_pixel_connections_to_hdf5(
        hdf5_path,
        group_path,
        connection_list):
    """
    Write a list of PixelSpaceConnections to a specific
    group in and HDF5 file
    """

    for conn in connection_list:
        if not isinstance(conn, PixelSpaceConnection):
            raise RuntimeError(
                "serialization requires connections to be of type "
                "PixelSpaceConnection; at least one of your "
                f"connections is of type {type(conn)}; "
                "Connections have a method self.to_pixel_space_connection "
                "to create an equivalent PixelSpaceConnection, if that "
                "helps."
            )

    with h5py.File(hdf5_path, 'a') as dst:
        if group_path in dst:
            raise RuntimeError(
                f"{group_path} arleady in {hdf5_path}; "
                "unclear how to proceed."
            )
        dst_grp = dst.create_group(group_path)

        dst_grp.create_dataset(
            "bezier_control_points",
            data=np.vstack(
                [conn.bezier_control_points
                 for conn in connection_list]
            )
        )

        dst_grp.create_dataset(
            "rendering_corners",
            data=np.vstack(
                [conn.rendering_corners
                 for conn in connection_list]
            )
        )

        dst_grp.create_dataset(
            "src_neighbor_fraction",
            data=np.array(
                [conn.src_neighbor_fraction
                 for conn in connection_list]
            )
        )

        dst_grp.create_dataset(
            "dst_neighbor_fraction",
            data=np.array(
                [conn.dst_neighbor_fraction
                 for conn in connection_list]
            )
        )

        dst_grp.create_dataset(
            "src_label",
            data=np.array(
                [conn.src_label.encode('utf-8')
                 for conn in connection_list]
            )
        )

        dst_grp.create_dataset(
            "dst_label",
            data=np.array(
                [conn.dst_label.encode('utf-8')
                 for conn in connection_list]
            )
        )


def read_pixel_connections_from_hdf5(
        hdf5_path,
        group_path,
        fov,
        convert_to_embedding=False):
    """
    Read a list of PixelSpaceConnections from a specific
    group in an HDF5 file. Return the list of
    PixelSpaceConnections
    """

    with h5py.File(hdf5_path, 'r') as src:
        result = read_pixel_connections_from_hdf5_handle(
            hdf5_handle=src,
            group_path=group_path,
            fov=fov,
            convert_to_embedding=convert_to_embedding)
    return result


def read_pixel_connections_from_hdf5_handle(
        hdf5_handle,
        group_path,
        fov,
        convert_to_embedding=False):

    src_grp = hdf5_handle[group_path]
    src_label_list = [
        label.decode('utf-8')
        for label in src_grp['src_label'][()]
    ]
    dst_label_list = [
        label.decode('utf-8')
        for label in src_grp['dst_label'][()]
    ]
    src_frac_list = src_grp['src_neighbor_fraction'][()]
    dst_frac_list = src_grp['dst_neighbor_fraction'][()]
    n_connections = len(src_label_list)

    rendering_corners = src_grp['rendering_corners'][()]

    if convert_to_embedding:
        rendering_corners = fov.transform_to_embedding_coordinates(
            rendering_corners
        )

    rendering_corners = rendering_corners.reshape(
        (n_connections, 4, 2)
    )

    bezier_control_points = src_grp['bezier_control_points'][()]

    if convert_to_embedding:
        bezier_control_points = fov.transform_to_embedding_coordinates(
            bezier_control_points
        )

    bezier_control_points=bezier_control_points.reshape(
        (n_connections, 2, 2)
    )

    result = [
        PixelSpaceConnection(
            src_label=src_label_list[ii],
            dst_label=dst_label_list[ii],
            src_neighbor_fraction=src_frac_list[ii],
            dst_neighbor_fraction=dst_frac_list[ii],
            rendering_corners=rendering_corners[ii, :, :],
            bezier_control_points=bezier_control_points[ii, :, :]
        )
        for ii in range(n_connections)
    ]
    return result


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
        self._thermal_control_point = None

    def to_pixel_space_connection(self):
        if not self.ready_to_render:
            raise RuntimeError(
                "Your connection is not ready to render\n"
                f"rendering_corners: {self.rendering_corners}\n"
                f"control points: {self.bezier_control_points}\n"
            )
        return PixelSpaceConnection(
            src_label=self.src.label,
            dst_label=self.dst.label,
            src_neighbor_fraction=self.src_neighbor_fraction,
            dst_neighbor_fraction=self.dst_neighbor_fraction,
            rendering_corners=self.rendering_corners,
            bezier_control_points=self.bezier_control_points
        )

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
    def thermal_control_point(self):
        return self._thermal_control_point

    @thermal_control_point.setter
    def thermal_control_point(self, value):
        assert self._thermal_control_point is None
        assert not hasattr(self, '_src_mid')
        self._thermal_control_point = value

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

        connection = dst_pt-self.thermal_control_point
        norm = np.sqrt((connection**2).sum())
        self._dst_mid = -self.dst.radius*connection/norm

        connection = self.thermal_control_point-src_pt
        norm = np.sqrt((connection**2).sum())
        self._src_mid = self.src.radius*connection/norm

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

        # this will trigger if one end of a connection is a point
        # (not sure what, if anything, I want to do about this
        # case)
        if geometry_utils.do_intersect([points[0], points[1]],
                                       [points[2], points[3]]):
            print(f'huh {self.src.label} {self.dst.label}')

    def set_bezier_control_points(self):
        """
        Thermal control is the result of the get_bezier_control_points
        function run on all the connections in the field of view
        """
        thermal_control = self.thermal_control_point
        assert self.rendering_corners is not None
        mid_pt = 0.5*(self.src.center_pt+self.src_mid+self.dst.center_pt+self.dst_mid)
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
        centroid_list,
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
    #end_charge = 5.0
    #mid_charge = 5.0
    #self_end_charge = 5.0
    spring_constant = 1.0
    time_step = 0.1

    n_conn = len(connection_list)
    n_centroids = len(centroid_list)
    n_background = len(centroid_list)
    background = np.zeros((n_centroids+n_conn, 2), dtype=float)
    orthogonals = np.zeros((n_conn, 2), dtype=float)
    distances = np.zeros(n_conn, dtype=float)
    charges = np.zeros(n_centroids+n_conn, dtype=float)
    origins = np.zeros((n_conn, 2), dtype=float)
    velocities = np.zeros((n_conn, 2), dtype=float)

    # if True, then the connection is linked to the centroid
    is_linked = np.zeros((n_conn, n_centroids), dtype=bool)

    for i0 in range(n_conn):
        conn = connection_list[i0]
        for i1 in range(n_centroids):
            if conn.src is centroid_list[i1]:
                is_linked[i0][i1] = True
            if conn.dst is centroid_list[i1]:
                is_linked[i0][i1] = True

    assert is_linked.sum() == 2*n_conn

    # first n_centroids points are the centroids
    for ii, centroid in enumerate(centroid_list):
        background[ii, :] = centroid.center_pt
        charges[ii] = centroid.radius

    mid_charge = 0.5*np.median(charges[:n_centroids])

    # then the bezier control points
    for i_conn, conn in enumerate(connection_list):
        background[n_centroids+i_conn, :] = 0.5*(conn.src.center_pt
                                              + conn.dst.center_pt)
        origins[i_conn, :] = 0.5*(conn.src.center_pt
                                  + conn.dst.center_pt)
        charges[n_centroids+i_conn] = mid_charge

        dd = conn.dst.center_pt-conn.src.center_pt
        distances[i_conn] = np.sqrt(
            (dd**2).sum()
        )
        dd = dd/distances[i_conn]
        orthogonals[i_conn, :] = geometry_utils.rot(dd, 0.5*np.pi)

    charges *= time_step
    spring_constant *= time_step

    max_acc = 100.0
    n_iter = 101

    # don't let a point drift more than this ratio times the distance
    # between the connection's end points away from its initial position
    max_total_displacement = 0.25

    n_tot = 0
    n_adj = 0
    spring_gt_coulomb = 0.0

    mask = np.ones(n_centroids+n_conn, dtype=bool)
    #functional_charges = np.copy(charges)
    keep_moving = np.ones(n_conn, dtype=bool)

    for i_iter in range(n_iter):
        if keep_moving.sum() == 0:
            break
    #while keep_moving.sum() > 0:
        for i_conn in range(n_conn):
            if not keep_moving[i_conn]:
                continue

            # mask out self repulsion
            mask[n_centroids+i_conn] = False

            # make each connection's own end points attractive
            is_linked_idx = np.where(is_linked[i_conn, :])[0]
            mask[is_linked_idx] = False
            #functional_charges[is_linked_idx] = self_end_charge

            test_pt = background[n_centroids+i_conn, :]
            coulomb_force = compute_coulomb_force(
                test_pt=test_pt,
                background_points=background[mask, :],
                charges=charges[mask],
                src_pt=background[is_linked_idx[0]],
                dst_pt=background[is_linked_idx[1]],
                n_centroids=n_centroids-2  # -2 because masking src and end pts
            )

            if True:
                from_origin = (test_pt-origins[i_conn, :])
                d_from_origin = np.sqrt((from_origin**2).sum())
                spring_force = -1.0*spring_constant*from_origin/(distances[i_conn]**2)
                force = coulomb_force + spring_force
                if (spring_force**2).sum() > (coulomb_force**2).sum():
                    spring_gt_coulomb += 1
            else:
                force = coulomb_force

            alpha = np.dot(force, orthogonals[i_conn, :])
            force = alpha*orthogonals[i_conn, :]

            #acc = np.sqrt((force**2).sum())
            #if acc > max_acc:
            #    force *= max_acc/acc
            #    n_adj += 1

            candidate = test_pt + velocities[i_conn, :]
            velocities[i_conn, :] += force
            dd = np.sqrt(((candidate-origins[i_conn, :])**2).sum())/distances[i_conn]
            if dd > max_total_displacement:
                keep_moving[i_conn] = False
            else:
                background[n_centroids+i_conn, :] = candidate

            mask[n_centroids+i_conn] = True
            mask[is_linked_idx] = True
            #functional_charges[is_linked_idx] = end_charge

            displacement = np.sqrt((velocities**2).sum())
            if displacement > 1.0e-3:
                n_tot += 1

        results = background[n_centroids:, :]
        dd = np.sqrt(((results-origins)**2).sum(axis=1))/distances
        print(f"    {n_tot} pts displaced -- adjusted {n_adj} accelerations -- "
              f"keep_moving {keep_moving.sum()} vs {keep_moving.shape} -- "
              f"spring_gt {spring_gt_coulomb} -- ddmax {dd.max()}")
    results = background[n_centroids:, :]
    return results


def compute_coulomb_force(
        test_pt,
        background_points,
        charges,
        src_pt,
        dst_pt,
        n_centroids,
        eps=0.001):

    bez = bezier_utils.quadratic_bezier(
        src_pt=src_pt,
        dst_pt=dst_pt,
        ctrl_pt=test_pt,
        t_steps=50
    )

    rsq = 1.0e6*np.ones(background_points.shape[0], dtype=float)
    vectors = np.zeros(background_points.shape, dtype=float)
    for bb in bez:
        delta = (bb-background_points[:n_centroids, :])
        delta_rsq = (delta**2).sum(axis=1)
        valid = np.where(delta_rsq < rsq[:n_centroids])[0]
        vectors[valid, :] = delta[valid, :]
        rsq[valid] = delta_rsq[valid]


    vectors[n_centroids:, :] = (test_pt-background_points[n_centroids:, :])
    rsq[n_centroids:] = (vectors[n_centroids:]**2).sum(axis=1)

    rsq = np.where(rsq > eps, rsq, eps)
    weights = charges/np.power(rsq, 1.5)
    force = (vectors.transpose()*weights).sum(axis=1)
    return force


class PixelSpaceConnection(object):

    def __init__(
            self,
            src_label,
            dst_label,
            src_neighbor_fraction,
            dst_neighbor_fraction,
            rendering_corners,
            bezier_control_points):
        """
        This is the form of the connection that actually
        gets rendered

        Parameters
        ----------
        src_label:
            label of the centroid that is the source for the
            connection
        dst_label:
            label of the centroid that is the destionation of
            the connection
        src_neighbor_fraction:
            fraction of src's neighbors that map to dst
        dst_neighbor_fraction:
            fraction of dst's neighbors that map to src
        rendering_corners:
            list of four 2-D points in pixel space that are the
            corners of the connection
        bezier_control_points:
            list of two 2-D points that are the control points for
            the Bezier curves of the connection
        """

        self._src_label = src_label
        self._dst_label = dst_label
        self._rendering_corners = rendering_corners
        self._bezier_control_points = bezier_control_points
        self._src_neighbor_fraction = src_neighbor_fraction
        self._dst_neighbor_fraction = dst_neighbor_fraction

    @property
    def src_label(self):
        return self._src_label

    @property
    def dst_label(self):
        return self._dst_label

    @property
    def src_neighbor_fraction(self):
        return self._src_neighbor_fraction

    @property
    def dst_neighbor_fraction(self):
        return self._dst_neighbor_fraction

    @property
    def rendering_corners(self):
        return self._rendering_corners

    @property
    def bezier_control_points(self):
        return self._bezier_control_points
