# -*- coding: utf-8 -*-
from __future__ import print_function

import math

import numpy as np
from pytest import approx

from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell_stable import EllStable


class MyQuasicvxOracle:
    def assess_optim(self, z, t: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            t (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        sqrtx, ly = z

        # constraint 1: exp(x) <= y, or sqrtx**2 <= ly
        if (fj := sqrtx * sqrtx - ly) > 0:
            return (np.array([2 * sqrtx, -1.0]), fj), None

        # constraint 3: x > 0
        # if x <= 0.:
        #     return (np.array([-1., 0.]), -x), None

        # objective: minimize -sqrt(x) / y
        tmp2 = math.exp(ly)
        tmp3 = t * tmp2
        if (fj := -sqrtx + tmp3) >= 0.0:  # feasible
            return (np.array([-1.0, tmp3]), fj), None
        t = sqrtx / tmp2
        return (np.array([-1.0, sqrtx]), 0), t


def test_case_feasible():
    """[summary]"""
    x0 = np.array([0.0, 0.0])  # initial x0
    E = EllStable(10.0, x0)
    P = MyQuasicvxOracle()
    xb, fb, _, _ = cutting_plane_optim(P, E, 0.0)
    assert xb is not None
    assert fb == approx(0.4288673396685956)
    assert xb[0] * xb[0] == approx(0.5029823096186075)
    assert math.exp(xb[1]) == approx(1.6536872634520428)


def test_case_infeasible1():
    """[summary]"""
    x0 = np.array([100.0, 100.0])  # wrong initial guess,
    E = EllStable(10.0, x0)  # or ellipsoid is too small
    P = MyQuasicvxOracle()
    xb, _, _, _ = cutting_plane_optim(P, E, 0.0)
    assert xb is None


def test_case_infeasible2():
    """[summary]"""
    x0 = np.array([0.0, 0.0])  # initial x0
    E = EllStable(10.0, x0)
    P = MyQuasicvxOracle()
    xb, _, _, _ = cutting_plane_optim(P, E, 100)  # wrong init best-so-far
    assert xb is None