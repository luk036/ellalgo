# -*- coding: utf-8 -*-
from __future__ import print_function

import math

import numpy as np
from pytest import approx

from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.cutting_plane import OracleOptim
from ellalgo.ell import Ell


class MyQuasicvxOracle(OracleOptim):
    def assess_optim(self, z, target: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            target (float): the best-so-far optimal value

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
        tmp3 = target * tmp2
        if (fj := -sqrtx + tmp3) >= 0.0:  # feasible
            return (np.array([-1.0, tmp3]), fj), None
        target = sqrtx / tmp2
        return (np.array([-1.0, sqrtx]), 0), target


def test_case_feasible():
    """[summary]"""
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, fbest, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    assert fbest == approx(0.4288673396685956)
    assert xbest[0] * xbest[0] == approx(0.5029823096186075)
    assert math.exp(xbest[1]) == approx(1.6536872634520428)


def test_case_infeasible1():
    """[summary]"""
    xinit = np.array([100.0, 100.0])  # wrong initial guess,
    ellip = Ell(10.0, xinit)  # or ellipsoid is too small
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is None


def test_case_infeasible2():
    """[summary]"""
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 100)  # wrong init best-so-far
    assert xbest is None
