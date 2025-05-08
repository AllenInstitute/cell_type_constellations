import h5py
import numpy as np
import scipy

import cell_type_constellations.utils.geometry_utils as geometry_utils


def merge_hulls(
        cell_set,
        visualization_coords,
        type_field,
        type_value,
        leaf_hull_path):

    leaf_list = cell_set.parent_to_leaves(
        type_field=type_field,
        type_value=type_value
    )

    if len(leaf_list) == 0:
        return []

    raw_hull_list = []
    with h5py.File(leaf_hull_path, 'r', swmr=True) as src:
        for leaf_value in leaf_list:
            if leaf_value not in src:
                continue
            leaf_grp = src[leaf_value]
            for idx in leaf_grp.keys():
                raw_hull_list.append(
                    scipy.spatial.ConvexHull(
                        leaf_grp[idx][()]
                    )
                )

    if len(raw_hull_list) == 0:
        return []

    raw_hull_list = winnow_hull_list(
        raw_hull_list,
        cutoff_quantile=0.01
    )

    raw_hull_list = [
        {'hull': hull}
        for hull in raw_hull_list
    ]

    data = get_pixellized_test_pts_from_type(
        cell_set=cell_set,
        visualization_coords=visualization_coords,
        type_field=type_field,
        type_value=type_value
    )

    test_pts = data['test_pts']
    test_pt_validity = data['test_pt_validity']

    keep_going = True
    final_pass = False
    min_overlap = 0.9
    min_f1 = 0.0
    nn_cutoff = 2
    while keep_going:
        centroid_array = np.array([
            _get_hull_centroid(h['hull']) for h in raw_hull_list
        ])

        area_array = np.array([
            h['hull'].volume for h in raw_hull_list
        ])

        dsq_array = geometry_utils.pairwise_distance_sq(centroid_array)
        if not final_pass:
            n_cols = nn_cutoff+1
            median_dsq = np.quantile(dsq_array[:, :n_cols], 0.25)

        mergers = dict()
        been_merged = set()
        skip_anyway = set()
        idx_list = np.argsort(area_array)[-1::-1]
        for i0 in idx_list:

            if i0 in been_merged or i0 in skip_anyway:
                continue

            sorted_i1 = np.argsort(dsq_array[i0, :])
            if not final_pass:
                sorted_i1 = sorted_i1[:nn_cutoff+1]
            for i1 in sorted_i1:
                if i1 == i0:
                    continue

                if i1 in been_merged or i1 in skip_anyway:
                    continue

                if not final_pass:
                    if dsq_array[i0, i1] > median_dsq:
                        continue

                new_hull = evaluate_merger(
                    raw_hull_list[i0],
                    raw_hull_list[i1],
                    test_pts=test_pts,
                    test_pt_validity=test_pt_validity,
                    min_overlap=min_overlap,
                    min_f1=min_f1)

                if new_hull is not None:
                    mergers[i0] = new_hull
                    been_merged.add(i0)
                    been_merged.add(i1)

                    if final_pass:
                        # do not further consider hulls
                        # who would have been nearest neighbors
                        # of this hull
                        for alt_i1 in sorted_i1[:nn_cutoff+1]:
                            skip_anyway.add(alt_i1)

                    break

        if len(mergers) == 0:
            if final_pass:
                return [h['hull'] for h in raw_hull_list]
            else:
                final_pass = True
                min_overlap = 1.1
                min_f1 = 0.99

        new_hull_list = []
        for ii in range(len(idx_list)):
            if ii not in been_merged:
                new_hull_list.append(raw_hull_list[ii])
            elif ii in mergers:
                new_hull_list.append(mergers[ii])
        raw_hull_list = new_hull_list
        if len(raw_hull_list) == 1:
            return [h['hull'] for h in raw_hull_list]


def get_pixellized_test_pts_from_type(
        cell_set,
        visualization_coords,
        type_field,
        type_value,
        min_res=0.01):

    data = get_test_pts_from_type(
        cell_set=cell_set,
        visualization_coords=visualization_coords,
        type_field=type_field,
        type_value=type_value)

    valid_pts = data['valid_pts']

    raw_test_pts = data['test_pts']
    raw_test_pt_validity = data['test_pt_validity']

    xmin = raw_test_pts[:, 0].min()
    xmax = raw_test_pts[:, 0].max()
    ymin = raw_test_pts[:, 1].min()
    ymax = raw_test_pts[:, 1].max()

    dd_res = 100000.0
    resx = dd_res
    resy = dd_res
    with np.errstate(invalid='ignore'):
        if valid_pts.shape[0] < 1000:
            dd = np.sqrt(geometry_utils.pairwise_distance_sq(valid_pts))
            dd_max = dd.max()
            idx_arr = np.arange(valid_pts.shape[0], dtype=int)
            dd[idx_arr, idx_arr] = dd_max
            dd_min = dd.min(axis=1)
            assert dd_min.shape == (valid_pts.shape[0], )
            del dd
            dd_res = max(min_res, np.median(dd_min))
        else:
            dd_res = 1000.0

    resx = max(min_res, (xmax-xmin)/100.0)
    resy = max(min_res, (ymax-ymin)/100.0)
    res = min(resx, resy, dd_res)

    resx = res
    resy = res

    nx = np.round(1+(xmax-xmin)/resx).astype(int)
    ny = np.round(1+(ymax-ymin)/resy).astype(int)

    grid = np.zeros((nx, ny), dtype=bool)
    valid_x = np.round((valid_pts[:, 0]-xmin)/resx).astype(int)
    valid_y = np.round((valid_pts[:, 1]-ymin)/resy).astype(int)
    grid[valid_x, valid_y] = True
    valid_idx = np.where(grid)

    valid_pts = np.array(
        [valid_idx[0]*resx+xmin,
         valid_idx[1]*resy+ymin]
    ).transpose()

    test_pts_tuple = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    test_pts = np.array([
        test_pts_tuple[0].flatten()*resx+xmin,
        test_pts_tuple[1].flatten()*resy+ymin
    ]).transpose()
    test_pt_validity = grid.flatten()

    # add back in the original invalid test points, so that those
    # pixels get up-weighted in false negative calculations

    test_pts_tuple = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    invalid_pts = raw_test_pts[np.logical_not(raw_test_pt_validity), :]
    grid = np.zeros((nx, ny), dtype=bool)
    invalid_x = np.round((invalid_pts[:, 0]-xmin)/resx).astype(int)
    invalid_y = np.round((invalid_pts[:, 1]-ymin)/resy).astype(int)

    # just mask indexes to fit in grid
    # (hacky; no idea why this is necessary)
    xy_mask = np.logical_and(
        invalid_x<nx,
        invalid_y<ny
    )
    invalid_x = invalid_x[xy_mask]
    invalid_y = invalid_y[xy_mask]

    grid[invalid_x, invalid_y] = True

    # if there is a valid point in that pixel, do not add it
    # as a double-counted invalid point
    grid[valid_x, valid_y] = False

    invalid_idx = np.where(grid)
    invalid_pts = np.array(
        [invalid_idx[0]*resx+xmin,
         invalid_idx[1]*resy+ymin]
    ).transpose()
    test_pts = np.concatenate([test_pts, invalid_pts])
    test_pt_validity = np.concatenate(
        [test_pt_validity, np.zeros(invalid_pts.shape[0], dtype=bool)]
    )

    return {
        'valid_pts': valid_pts,
        'test_pts': test_pts,
        'test_pt_validity': test_pt_validity
    }


def get_test_pts_from_type(
        cell_set,
        visualization_coords,
        type_field,
        type_value):

    valid_pt_idx = cell_set.type_mask(
        type_field=type_field,
        type_value=type_value
    )
    valid_pts = visualization_coords[valid_pt_idx, :]

    xmin = valid_pts[:, 0].min()
    xmax = valid_pts[:, 0].max()
    ymin = valid_pts[:, 1].min()
    ymax = valid_pts[:, 1].max()
    dx = xmax-xmin
    dy = ymax-ymin
    slop = 0.05
    xmin -= slop*dx
    xmax += slop*dx
    ymin -= slop*dy
    ymax += slop*dy

    test_pt_mask = np.logical_and(
        visualization_coords[:, 0] > xmin,
        np.logical_and(
            visualization_coords[:, 0] < xmax,
            np.logical_and(
                visualization_coords[:, 1] > ymin,
                visualization_coords[:, 1] < ymax
            )
        )
    )
    test_pts = visualization_coords[test_pt_mask, :]

    test_pt_idx = np.where(test_pt_mask)[0]
    valid_pt_idx = set(valid_pt_idx)
    test_pt_validity = np.array([
        ii in valid_pt_idx for ii in test_pt_idx
    ])
    assert test_pt_validity.sum() == len(valid_pt_idx)
    return {
        'valid_pts': valid_pts,
        'test_pts': test_pts,
        'test_pt_validity': test_pt_validity
    }


def _get_hull_centroid(hull):
    return np.mean(hull.points, axis=0)


def evaluate_merger(
        hull0,
        hull1,
        test_pts,
        test_pt_validity,
        min_overlap=0.9,
        min_f1=0.0):

    new_hull = scipy.spatial.ConvexHull(
        np.concatenate([hull0['hull'].points, hull1['hull'].points])
    )

    if 'in' not in hull0:
        hull0['in'] = pts_in_hull(
            hull=hull0['hull'],
            pts=test_pts
        )
    in0 = hull0['in']

    if 'in' not in hull1:
        hull1['in'] = pts_in_hull(
            hull=hull1['hull'],
            pts=test_pts
        )
    in1 = hull1['in']

    overlap = np.logical_and(in0, in1).sum()

    n1 = in1.sum()
    n0 = in0.sum()
    min_n = min(n0, n1)

    if overlap > 0:
        if overlap > min_overlap*min_n:
            return {'hull': new_hull}

    in_new = pts_in_hull(
        hull=new_hull,
        pts=test_pts)

    in_all = np.logical_or(
        in_new,
        np.logical_or(
            in0,
            in1
        )
    )

    tp0 = np.logical_and(
        in0,
        test_pt_validity
    )

    tp1 = np.logical_and(
        in1,
        test_pt_validity
    )

    tp_old = np.logical_or(tp0, tp1).sum()

    false_points = np.logical_not(test_pt_validity)

    fp_new = np.logical_and(
        in_new,
        false_points
    ).sum()
    tp_new = np.logical_and(
        in_new,
        test_pt_validity
    ).sum()

    if tp_new == 0:
        return None

    fn_new = np.logical_and(
        in_all,
        np.logical_and(
            np.logical_not(in_new),
            test_pt_validity
        )
    ).sum()
    f1_new = tp_new/(tp_new+0.5*(fn_new+fp_new))

    in_old = np.logical_or(in0, in1)

    fp_old = np.logical_and(
        in_old,
        false_points
    ).sum()

    fn_old = np.logical_and(
        in_all,
        np.logical_and(
            np.logical_not(in_old),
            test_pt_validity
        )
    ).sum()
    f1_old = tp_old/(tp_old+0.5*(fn_old+fp_old))

    if f1_new > f1_old and f1_new > min_f1:
        return {'hull': new_hull, 'in': in_new}

    return None


def winnow_hull_list(
        hull_list,
        cutoff_quantile=0.05):
    """
    Take a list of ConvexHulls;
    Winnow by n-weighted density (n is the number of
    points in the hull);
    Return list of ConvexHulls that just contain vertices.
    """

    assert len(hull_list) > 0
    density_list = []
    for hull in hull_list:
        density_list += [density_from_hull(hull)]*hull.points.shape[0]
    cutoff = np.quantile(density_list, cutoff_quantile)
    result = [
        scipy.spatial.ConvexHull(hull.points[hull.vertices, :])
        for hull in hull_list
        if density_from_hull(hull) >= cutoff
    ]
    return result


def density_from_hull(hull):
    return hull.points.shape[0]/hull.volume


def pts_in_hull(pts, hull):
    """
    Points on the hull edge are not considered
    "in" the hull
    """
    n_vert = len(hull.vertices)

    sgn_arr = None
    result = np.ones(pts.shape[0], dtype=bool)
    result_idx = np.arange(pts.shape[0], dtype=int)
    for ii in range(n_vert):

        (pt_vec,
         edge) = _get_pt_edge(
                     hull=hull,
                     ii=ii,
                     n_vert=n_vert,
                     pts=pts,
                     result=result)

        sgn = np.sign(geometry_utils.cross_product_2d_bulk(
                            vec0=pt_vec,
                            vec1=edge)
                      ).astype(int)

        if sgn_arr is None:
            sgn_arr = sgn[:, 0]
        else:
            (result,
             pts,
             sgn_arr,
             result_idx) = _update_result(
                sgn_arr=sgn_arr,
                sgn=sgn,
                result=result,
                result_idx=result_idx,
                pts=pts)

        if result.sum() == 0:
            break

    return result


def _update_result(sgn_arr, sgn, result, result_idx, pts):
    invalid = (sgn_arr != sgn[:, 0])
    valid = np.logical_not(invalid)
    result[result_idx[invalid]] = False
    result_idx = result_idx[valid]
    sgn_arr = sgn_arr[valid]
    pts = pts[valid, :]
    return result, pts, sgn_arr, result_idx


def _get_pt_edge(
        hull,
        ii,
        n_vert,
        pts,
        result):

    src = hull.points[hull.vertices[ii]]
    i1 = ii+1
    if i1 >= n_vert:
        i1 = 0
    dst = hull.points[hull.vertices[i1]]
    edge = np.array([dst-src])
    pt_vec = pts-src
    return pt_vec, edge
