import h5py
import multiprocessing
import numpy as np
import os
import pathlib
import scipy
import tempfile
import time

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up
)

from cell_type_mapper.utils.multiprocessing_utils import (
    winnow_process_list
)

import cell_type_constellations.utils.str_utils as str_utils
import cell_type_constellations.utils.geometry_utils as geometry_utils
import cell_type_constellations.hulls.classes as hull_classes
import cell_type_constellations.hulls.leaf_utils as leaf_utils
import cell_type_constellations.hulls.merger_utils as merger_utils


def create_and_serialize_all_hulls(
        cell_set,
        visualization_coords,
        fov,
        dst_path,
        n_processors=4,
        tmp_dir=None):
    """
    Instantiate a list of PixelSpaceHulls and write them to an
    HDF5 file.

    Parameters
    ----------
    cell_set:
        the CellSet defining the taxonomy
    visualization_coords:
        the (n_cells, 2) array of coordinates defining the
        visualization
    fov:
        the FieldOfView defining the visualization
    dst_path:
        path to the HDF5 file where output will be written
    n_processors:
        number of independent worker processes to spin up.
    tmp_dir:
        path to directory where scratch files can be written
    """

    print('=======CREATING HULLS=======')

    tmp_dir = tempfile.mkdtemp(
        dir=tmp_dir,
        prefix='hull_scratch_')
    try:
        _create_and_serialize_all_hulls(
            cell_set=cell_set,
            visualization_coords=visualization_coords,
            fov=fov,
            dst_path=dst_path,
            n_processors=n_processors,
            tmp_dir=tmp_dir
        )
    finally:
        _clean_up(tmp_dir)


def _create_and_serialize_all_hulls(
        cell_set,
        visualization_coords,
        fov,
        dst_path,
        n_processors,
        tmp_dir):

    t0 = time.time()
    dst_path = pathlib.Path(dst_path)

    leaf_hull_path = mkstemp_clean(
        dir=tmp_dir,
        prefix='leaf_hull_cache_',
        suffix='.h5'
    )

    leaf_utils.get_all_leaf_hulls(
        cell_set=cell_set,
        visualization_coords=visualization_coords,
        dst_path=leaf_hull_path,
        clobber=True
    )

    print("=======CREATED LEAF HULL CACHE=======")

    # assemble an array of valid type_field, type_value pairs
    # along with an array of how many cells are associated
    # with each pair, so we can try to distribute the work
    # evenly

    type_pair_list = []
    n_cells_list = []
    for type_field in cell_set.type_field_list():
        for type_value in cell_set.type_value_list(type_field):
            leaves = cell_set.parent_to_leaves(
                type_field,
                type_value
            )
            if len(leaves) == 0:
                continue
            type_pair_list.append((type_field, type_value))
            n_cells_list.append(cell_set.n_cells_in_type(
                                    type_field=type_field,
                                    type_value=type_value))

    sorted_idx = np.argsort(n_cells_list)[-1::-1]
    sub_lists = []
    for ii in range(2*n_processors):
        sub_lists.append([])

    for ii, idx in enumerate(sorted_idx):
        i_sub = ii % len(sub_lists)
        sub_lists[i_sub].append(type_pair_list[idx])

    process_list = []
    mgr = multiprocessing.Manager()
    lock = mgr.Lock()

    print("=======STARTING HULL SERIALIZATION=======")
    for type_field_value_list in sub_lists:
        p = multiprocessing.Process(
            target=create_and_serialize_pixel_hull_list,
            kwargs={
                'cell_set': cell_set,
                'visualization_coords': visualization_coords,
                'fov': fov,
                'type_field_value_list': type_field_value_list,
                'leaf_hull_path': leaf_hull_path,
                'dst_path': dst_path,
                'lock': lock
            }
        )
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = winnow_process_list(process_list)

    while len(process_list) > 0:
        process_list = winnow_process_list(process_list)

    dur = (time.time()-t0)/60.0
    print(f'=======WROTE HULLS TO {dst_path} in {dur:.2e} minutes=======')


def create_and_serialize_pixel_hull_list(
        cell_set,
        visualization_coords,
        fov,
        type_field_value_list,
        leaf_hull_path,
        dst_path,
        lock):
    """
    Write create a list of PixelSpaceHulls and write them to an
    HDF5 file.

    Parameters
    ----------
    cell_set:
        the CellSet defining the taxonomy
    visualization_coords:
        the (n_cells, 2) array of coordinates defining the
        visualization
    fov:
        the FieldOfView defining the visualization
    type_field_value_list:
        list of (type_field, type_value) pairs indicating which
        hulls to create and write
    leaf_hull_path:
        path to the HDF5 file containing the leaf hulls
        for this CellSet, visualization_coords pair
    dst_path:
        path to the HDF5 file where output will be written
    lock:
        a multiprocessing Lock to prevent multiple workers
        from writing to dst_path at once
    """
    t0 = time.time()
    hull_list = []
    for pair in type_field_value_list:
        type_field = pair[0]
        type_value = pair[1]
        bare_hull_list = load_single_hull(
                cell_set=cell_set,
                visualization_coords=visualization_coords,
                type_field=type_field,
                type_value=type_value,
                leaf_hull_path=leaf_hull_path
        )

        hull_list.append(
            hull_classes.PixelSpaceHull.from_bare_hull_list(
                bare_hull_list=bare_hull_list,
                fov=fov
            )
        )

    with lock:
        with h5py.File(dst_path, 'a') as dst:
            for pair, pixel_space_hull in zip(type_field_value_list,
                                              hull_list):
                type_field = pair[0]
                type_value = pair[1]
                safe_type_value = str_utils.sanitize_taxon_name(type_value)
                group_path = f'hulls/{type_field}/{safe_type_value}'
                pixel_space_hull.to_hdf5_handle(
                    hdf5_handle=dst,
                    group_path=group_path
                )

    dur = (time.time()-t0)/60.0
    n = len(type_field_value_list)
    print(
        f'    worker process {os.getpid()} wrote '
        f'{n} hulls in {dur:.2e} minutes'
    )


def load_single_hull(
        cell_set,
        visualization_coords,
        type_field,
        type_value,
        leaf_hull_path):

    bare_hull_list = [
        hull_classes.BareHull.from_convex_hull(h)
        for h in merger_utils.merge_hulls(
            cell_set=cell_set,
            visualization_coords=visualization_coords,
            type_field=type_field,
            type_value=type_value,
            leaf_hull_path=leaf_hull_path
        )
    ]

    if type_field == cell_set.leaf_type:
        return bare_hull_list

    return create_compound_bare_hull(
        bare_hull_list=bare_hull_list)


def create_compound_bare_hull(
        bare_hull_list):

    to_keep = []
    for i0 in range(len(bare_hull_list)):
        b0 = bare_hull_list[i0]
        found_match = False
        for i1 in range(i0+1, len(bare_hull_list), 1):
            b1 = bare_hull_list[i1]
            if _are_bare_hulls_identical(b0, b1):
                found_match = True
                break
        if not found_match:
            to_keep.append(b0)
    bare_hull_list = to_keep

    while True:
        new_hull = None
        n0 = len(bare_hull_list)
        has_merged = set()
        for i0 in range(len(bare_hull_list)):
            if len(has_merged) > 0:
                break
            h0 = bare_hull_list[i0]
            for i1 in range(i0+1, len(bare_hull_list), 1):
                h1 = bare_hull_list[i1]
                merger = merge_bare_hulls(h0, h1)
                if len(merger) == 1:
                    new_hull = merger[0]
                    has_merged.add(i0)
                    has_merged.add(i1)
                    break
        new_hull_list = []
        if new_hull is not None:
            new_hull_list.append(new_hull)
        for ii in range(len(bare_hull_list)):
            if ii not in has_merged:
                new_hull_list.append(bare_hull_list[ii])
        bare_hull_list = new_hull_list
        if len(bare_hull_list) == n0:
            break

    if len(bare_hull_list) == 0:
        return None

    return bare_hull_list


def merge_bare_hulls(
            bare0,
            bare1):
    convex0 = scipy.spatial.ConvexHull(bare0.points)
    convex1 = scipy.spatial.ConvexHull(bare1.points)

    # find all intersections between the segments in the two
    # bare hulls
    bare0_to_1 = dict()
    bare1_to_0 = dict()
    intersection_points = []
    n_all = bare0.points.shape[0]+bare1.points.shape[0]
    for i0, seg0 in enumerate(bare0.segments):
        for i1, seg1 in enumerate(bare1.segments):
            intersection = None
            if _are_segments_identical(seg0, seg1):
                intersection = 0.5*(seg0[0]+seg0[1])
            elif np.allclose(seg0[0], seg1[1], atol=0.0, rtol=1.0e-4):
                intersection = seg1[1]
            else:
                intersection = geometry_utils.find_intersection_pt(
                    seg0,
                    seg1)
            if intersection is not None:
                if i0 not in bare0_to_1:
                    bare0_to_1[i0] = dict()
                if i1 not in bare1_to_0:
                    bare1_to_0[i1] = dict()
                intersection_points.append(intersection)
                bare0_to_1[i0][i1] = n_all+len(intersection_points)-1
                bare1_to_0[i1][i0] = n_all+len(intersection_points)-1

    intersection_points = np.array(intersection_points)
    n_intersections = intersection_points.shape[0]

    if n_intersections < 2:

        bare1_in_0 = merger_utils.pts_in_hull(
            pts=bare1.points,
            hull=convex0
        )

        if bare1_in_0.all():
            return [bare0]

        bare0_in_1 = merger_utils.pts_in_hull(
            pts=bare0.points,
            hull=convex1)

        if bare0_in_1.all():
            return [bare1]

        return [bare0, bare1]

    all_points = np.concatenate([
        bare0.points,
        bare1.points,
        intersection_points])

    meta_intersection_lookup = [bare0_to_1, bare1_to_0]

    # For each bare hull, assemble a new ordered list of points that includes
    # the intersections.

    new_bare_hulls = []
    for i_hull, src_hull in enumerate([bare0, bare1]):
        new_hull = dict()   # a src->dst graph
        dst_set = set()
        if i_hull == 0:
            hull_offset = 0
        else:
            hull_offset = bare0.points.shape[0]
        intersection_lookup = meta_intersection_lookup[i_hull]
        for i_seg in src_hull.i_segments:
            src_idx = int(hull_offset + i_seg[0])

            src_pt = src_hull.points[i_seg[0]]
            if i_seg[0] in intersection_lookup:
                intersection_idx_list = np.array(
                    [idx for idx in intersection_lookup[i_seg[0]].values()]
                )
                ddsq_arr = np.array([
                    ((all_points[idx]-src_pt)**2).sum()
                    for idx in intersection_idx_list
                ])
                sorted_dex = np.argsort(ddsq_arr)
                intersection_idx_list = intersection_idx_list[sorted_dex]
                for dst_idx in intersection_idx_list:
                    dst_idx = int(dst_idx)

                    assert src_idx not in new_hull
                    assert dst_idx not in dst_set

                    new_hull[src_idx] = dst_idx
                    dst_set.add(dst_idx)
                    src_idx = dst_idx

            dst_idx = int(hull_offset + i_seg[1])

            assert src_idx not in new_hull
            new_hull[src_idx] = dst_idx
            assert dst_idx not in dst_set
            dst_set.add(dst_idx)

        new_bare_hulls.append(new_hull)

    # find a starting point that is on the outer perimeter of the
    # ConvexHull created from all_points (to guard against accidentally
    # staring on a vertex that is part of a hole in the merged
    # hull)
    naive_hull = scipy.spatial.ConvexHull(all_points)
    starting_idx = None
    for vertex in naive_hull.vertices:
        if vertex >= bare0.points.shape[0]:
            continue
        starting_idx = vertex
        break

    assert starting_idx is not None

    current_hull = 0
    final_points = [starting_idx]
    final_set = set(final_points)
    while True:
        next_pt = new_bare_hulls[current_hull][final_points[-1]]
        if next_pt == final_points[0]:
            break
        final_points.append(next_pt)
        final_set.add(next_pt)
        if next_pt in new_bare_hulls[(current_hull+1) % 2]:
            current_hull = ((current_hull+1) % 2)

    return [
        hull_classes.BareHull(
            points=np.array(
             [
              all_points[idx]
              for idx in final_points
             ]
            )
        )
    ]


def _are_segments_identical(seg0, seg1):
    for p0 in seg0:
        this_identical = False
        for p1 in seg1:
            if np.allclose(p1, p0, atol=0.0, rtol=1.0e-4):
                this_identical = True
        if not this_identical:
            return False
    return True


def _are_bare_hulls_identical(b0, b1):
    if b0.points.shape != b1.points.shape:
        return False
    points1 = [
        b1.points[ii, :]
        for ii in range(b1.points.shape[0])
    ]

    for ii in range(b0.points.shape[0]):
        p0 = b0.points[ii, :]
        found_it = False
        i_found = None
        for i1, p1 in enumerate(points1):
            if np.allclose(p0, p1, atol=0.0, rtol=1.0e-6):
                found_it = True
                i_found = i1
        if not found_it:
            return False
        points1.pop(i_found)
    return True
