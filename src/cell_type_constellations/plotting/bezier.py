"""
Functions to create the flavors of Bezier curves we use
https://en.wikipedia.org/wiki/B%C3%A9zier_curve
"""

import numpy as np


def quadratic_bezier(
        src_pt,
        dst_pt,
        ctrl_pt,
        t_steps=50):
    """
    All inputs are (2,) ndarrays representing the
    source point, destination point, and control point.

    Return a (N, 2) array of points along the curve
    """
    tvals = np.linspace(0, 1, t_steps).reshape(t_steps, 1)
    src_pt = src_pt.reshape((2,1)).transpose()
    dst_pt = dst_pt.reshape((2,1)).transpose()
    ctrl_pt = ctrl_pt.reshape((2,1)).transpose()

    pts = (
        np.dot((1.0-tvals)**2, src_pt)
        + 2.0*np.dot((1.0-tvals)*tvals, ctrl_pt)
        + np.dot(tvals**2, dst_pt)
    )
    return pts


def quadratic_ctrl_from_mid_pt(
        src_pt,
        mid_pt,
        dst_pt):
    """
    Find the control point for a bezier curve
    whose end points and mid point are given
    (mid point defined as the t=0.5 point)
    """
    pt = 2.0*(mid_pt-0.25*(src_pt+dst_pt))
    return pt

def cubic_bezier(
        src_pt,
        ctrl_pt0,
        ctrl_pt1,
        dst_pt,
        t_steps=50):
    """
    All inputs are (2,) ndarrays representing the
    source point, destination point, and control point.

    Return a (N, 2) array of points along the curve
    """
    tvals = np.linspace(0, 1, t_steps).reshape(t_steps, 1)
    src_pt = src_pt.reshape((2,1)).transpose()
    dst_pt = dst_pt.reshape((2,1)).transpose()
    ctrl_pt0 = ctrl_pt0.reshape((2,1)).transpose()
    ctrl_pt1 = ctrl_pt1.reshape((2,1)).transpose()

    pts = (
        np.dot((1.0-tvals)**3, src_pt)
        + 3.0*np.dot(tvals*(1.0-tvals)**2, ctrl_pt0)
        + 3.0*np.dot((1.0-tvals)*tvals**2, ctrl_pt1)
        + np.dot(tvals**3, dst_pt)
    )

    return pts
