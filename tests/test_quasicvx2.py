# -*- coding: utf-8 -*-
from __future__ import print_function

import math

import numpy as np
from pytest import approx

from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell


class MyQuasicvxOracle:
    def assess_optim(self, z, t: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            t (float): the best-so-far optimal value

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
        if (fj := -tmp2 - t * y) >= 0.0:  # infeasible
            return (np.array([-0.5 / tmp2, -t]), fj), None
        t = -tmp2 / y
        return (np.array([-0.5 / tmp2, -t]), 0), t


def test_case_feasible():
    """[summary]"""
    x0 = np.array([1.0, 1.0])  # initial x0
    E = Ell(10.0, x0)
    P = MyQuasicvxOracle()
    xb, fb, _, _ = cutting_plane_optim(P, E, 0.0)
    assert xb is not None
    assert fb == approx(-0.4288673396685956)
    assert xb[0] == approx(0.5053830040042219)
    assert xb[1] == approx(1.6576289475891712)


def test_case_infeasible1():
    """[summary]"""
    x0 = np.array([100.0, 100.0])  # wrong initial guess,
    E = Ell(10.0, x0)  # or ellipsoid is too small
    P = MyQuasicvxOracle()
    xb, _, _,_ = cutting_plane_optim(P, E, 0.0)
    assert xb is None


def test_case_infeasible2():
    """[summary]"""
    x0 = np.array([1.0, 1.0])  # initial x0
    E = Ell(10.0, x0)
    P = MyQuasicvxOracle()
    xb, _, _, _ = cutting_plane_optim(P, E, -100)  # wrong init best-so-far
    assert xb is None
