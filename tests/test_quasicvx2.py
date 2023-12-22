# -*- coding: utf-8 -*-
from __future__ import print_function

import math

import numpy as np
from pytest import approx

from ellalgo.cutting_plane import OracleOptim, cutting_plane_optim
from ellalgo.ell import Ell


class MyQuasicvxOracle(OracleOptim):
    def assess_optim(self, z, gamma: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            gamma (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        x, y = z

        # constraint 1: exp(x) <= y
        tmp = math.exp(x)
        if (fj := tmp - y) > 0.0:
            return (np.array([tmp, -1.0]), fj), None

        # constraint 2: y > 0
        if y <= 0.0:
            return (np.array([0.0, -1.0]), -y), None

        # constraint 3: x > 0
        if x <= 0.0:
            return (np.array([-1.0, 0.0]), -x), None

        # objective: minimize -sqrt(x) / y
        tmp2 = math.sqrt(x)
        if (fj := -tmp2 - gamma * y) >= 0.0:  # infeasible
            return (np.array([-0.5 / tmp2, -gamma]), fj), None
        gamma = -tmp2 / y
        return (np.array([-0.5 / tmp2, -gamma]), 0), gamma


def test_case_feasible():
    """[summary]"""
    xinit = np.array([1.0, 1.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, fbest, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    assert fbest == approx(-0.42888194247600586)
    assert xbest[0] == approx(0.5000004646814299)
    assert xbest[1] == approx(1.6487220368468205)


def test_case_infeasible1():
    """[summary]"""
    xinit = np.array([100.0, 100.0])  # wrong initial guess,
    ellip = Ell(10.0, xinit)  # or ellipsoid is too small
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is None


def test_case_infeasible2():
    """[summary]"""
    xinit = np.array([1.0, 1.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, -100)  # wrong init best-so-far
    assert xbest is None
