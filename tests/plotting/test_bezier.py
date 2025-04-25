import pytest

import numpy as np

import cell_type_constellations.plotting.bezier as bezier_utils


@pytest.mark.parametrize("t_steps", [5, 10, 25])
def test_quadratic_bezier(t_steps):

    src_pt = np.array([-1.0, 0.0])
    ctrl_pt = np.array([1.0, 2.0])
    dst_pt = np.array([-3.0, 10.0])

    actual = bezier_utils.quadratic_bezier(
        src_pt=src_pt,
        dst_pt=dst_pt,
        ctrl_pt=ctrl_pt,
        t_steps=t_steps
    )

    tvals = np.linspace(0, 1, t_steps)
    expected = np.zeros((t_steps, 2))
    for ii in range(t_steps):
        for jj in range(2):
            expected[ii][jj] = (
                src_pt[jj]*(1.0-tvals[ii])**2
                + ctrl_pt[jj]*2.0*(1.0-tvals[ii])*tvals[ii]
                + dst_pt[jj]*tvals[ii]**2
            )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=1.0e-6
    )


@pytest.mark.parametrize("t_steps", [5, 10, 25])
def test_cubic_bezier(t_steps):

    src_pt = np.array([-1.0, 0.0])
    ctrl_pt0 = np.array([1.0, 2.0])
    ctrl_pt1 = np.array([-2.0, 13.0])
    dst_pt = np.array([-3.0, 10.0])

    actual = bezier_utils.cubic_bezier(
        src_pt=src_pt,
        ctrl_pt0=ctrl_pt0,
        ctrl_pt1=ctrl_pt1,
        dst_pt=dst_pt,
        t_steps=t_steps
    )

    tvals = np.linspace(0, 1, t_steps)
    expected = np.zeros((t_steps, 2))
    for ii in range(t_steps):
        for jj in range(2):
            expected[ii][jj] = (
                src_pt[jj]*(1.0-tvals[ii])**3
                + ctrl_pt0[jj]*3.0*tvals[ii]*(1.0-tvals[ii])**2
                + ctrl_pt1[jj]*3.0*(1.0-tvals[ii])*tvals[ii]**2
                + dst_pt[jj]*tvals[ii]**3
            )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=0.0,
        rtol=1.0e-6
    )
