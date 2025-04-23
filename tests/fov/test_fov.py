import pytest

import numpy as np

import cell_type_constellations.visual_elements.fov as fov_utils


@pytest.mark.parametrize(
    "fov_height,x_min,x_max,y_min,y_max",
    [(1080, -10.0, 20.0, 100.0, 350.0),
     (977, -10.0, 20.0, 100.0, 350.0),
     (5.0, -20.0, 50.0, -50.0, 200.0)
     ]
)
def test_fov_coordinate_transformations(
        fov_height,
        x_min,
        x_max,
        y_min,
        y_max):
    """
    Do some simple roundtrip tests on the field of view
    embedding_to_pixel and pixel_to_embedding coordinate
    transformations.
    """

    rng = np.random.default_rng(223131)
    n_basis = 100
    basis_coords = rng.random((n_basis, 2))
    basis_coords[:, 0] = x_min + (x_max-x_min)*basis_coords[:, 0]
    basis_coords[:, 1] = y_min + (y_max-y_min)*basis_coords[:, 1]

    fov = fov_utils.FieldOfView.from_coords(
        coords=basis_coords,
        fov_height=fov_height,
        max_radius=20,
        min_radius=5
    )

    n_query = 20
    query_coords = rng.random((n_query, 2))
    query_coords[:, 0] = (x_min-10.0) + (x_max-x_min+20.0)*query_coords[:, 0]
    query_coords[:, 1] = (y_min-10.0) + (y_max-y_min+20.0)*query_coords[:, 0]

    pixel_coords = fov.transform_to_pixel_coordinates(query_coords)
    assert np.isfinite(pixel_coords).sum() == pixel_coords.size
    roundtrip = fov.transform_to_embedding_coordinates(pixel_coords)
    np.testing.assert_allclose(
        query_coords,
        roundtrip,
        atol=0.0,
        rtol=1.0e-6
    )
