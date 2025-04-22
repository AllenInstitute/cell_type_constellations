import numpy as np

from cell_type_constellations.utils.geometry_utils import (
    do_intersect,
    _do_overlap,
    cross_product_2d_bulk,
    cross_product_2d,
    find_intersection_pt
)


def test_do_overlap():
    assert _do_overlap(0.5, 1.0, 0.75, 2.0)
    assert _do_overlap(0.75, 2.0, 0.5, 1.0)
    assert _do_overlap(0.5, 1.0, 0.75, 0.8)
    assert _do_overlap(0.5, 1.0, 0.0, 2.0)
    assert not _do_overlap(-1, 0.5, 1, 2)


def test_do_intersect():

    pt1 = np.array([1, 1])
    pt2 = np.array([2, 2])
    pt3 = np.array([1.5, 1.5])
    pt4 = np.array([-1, -1])
    pt5 = np.array([0.5, 0.5])
    pt6 = np.array([2, 1])
    pt7 = np.array([1, 2])
    pt8 = np.array([2, 4])

    assert do_intersect([pt1, pt2], [pt3, pt4])
    assert do_intersect([pt3, pt4], [pt1, pt2])

    assert do_intersect([pt1, pt2], [pt4, pt3])
    assert do_intersect([pt4, pt3], [pt1, pt2])

    assert do_intersect([pt1, pt2], [pt6, pt7])
    assert do_intersect([pt6, pt7], [pt1, pt2])

    assert do_intersect([pt1, pt2], [pt6, pt8])
    assert do_intersect([pt6, pt8], [pt1, pt2])

    assert not do_intersect([pt1, pt2], [pt4, pt5])
    assert not do_intersect([pt1, pt2], [pt7, pt8])

    assert not do_intersect([pt4, pt5], [pt1, pt2])
    assert not do_intersect([pt7, pt8], [pt1, pt2])

    assert do_intersect([pt1, pt6], [pt1, pt2])
    assert do_intersect([pt1, pt2], [pt1, pt6])


def test_cross_product_bulk():

    vec0 = np.array([
        [0.2, 0.3],
        [1.1, 2.2],
        [3.1, 4.2]
    ])
    assert vec0.shape == (3, 2)

    vec1 = np.array([
        [-0.1, 0.9],
        [2.3, -3.4]
    ])

    actual = cross_product_2d_bulk(vec0, vec1)
    assert actual.shape == (3, 2)
    for ii in range(3):
        for jj in range(2):
            np.testing.assert_allclose(
                actual[ii, jj],
                cross_product_2d(vec0[ii, :], vec1[jj, :])[-1]
            )


def test_find_intersection_point():

    expected = np.array([2.3, 4.1])
    v0 = np.array([0.1, 0.3])
    v1 = np.array([-1.2, 0.4])

    segment0 = [expected+v0, expected-2.0*v0]
    segment1 = [expected-3.0*v1, expected + 0.5*v1]
    actual = find_intersection_pt(segment0, segment1)
    np.testing.assert_allclose(expected, actual, atol=0.0, rtol=1.0e-6)

    actual = find_intersection_pt(segment1, segment0)
    np.testing.assert_allclose(expected, actual, atol=0.0, rtol=1.0e-6)

    segment0 = np.array([expected, expected-v0])
    segment1 = np.array([expected-v1, expected+v1])

    assert do_intersect(segment0, segment1)

    actual = find_intersection_pt(segment0, segment1)
    np.testing.assert_allclose(expected, actual, atol=0.0, rtol=1.0e-6)

    assert do_intersect(segment1, segment0)
    actual = find_intersection_pt(segment1, segment0)
    np.testing.assert_allclose(expected, actual, atol=0.0, rtol=1.0e-6)

    segment0 = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]
    segment1 = [np.array([0.0, 4.0]), np.array([1.0, 4.0])]
    assert find_intersection_pt(segment0, segment1) is None

    segment0 = [np.array([0.0, 1.0]), np.array([1.0, 2.0])]
    segment1 = [np.array([0.5, 1.5]), np.array([1.5, 2.5])]
    assert find_intersection_pt(segment0, segment1) is None
