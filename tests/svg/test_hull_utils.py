import numpy as np
import scipy.spatial

from cell_type_constellations.hulls.merger_utils import (
    pts_in_hull
)


def test_points_in_hull():

    hull_points = np.array(
        [[0, 0],
         [0, 1],
         [0.5, 0.5],
         [1, 1],
         [1, 0],
         [0.5, 2],
         [0.5, 1.5]
         ]
    )
    hull = scipy.spatial.ConvexHull(hull_points)
    expected_vertices = [0, 1, 3, 4, 5]
    assert set(expected_vertices) == set(hull.vertices)

    test_points = np.array([
        [0.25, 0.25],
        [3.0, 0.1],
        [0.5, 1.25],
        [0.55, 1.2],
        [0.5, 3.0],
        [0.5, 2.0]  # points on hull edge are not considered "in" the hull
    ])

    is_in = pts_in_hull(pts=test_points, hull=hull)
    expected = np.array(
        [True, False, True, True, False, False]
    )
    np.testing.assert_array_equal(is_in, expected)
