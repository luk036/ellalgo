# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import CutStatus, cutting_plane_optim
from ellalgo.ell_stable import ell_stable


class MyOracle2:
    def assess_feas(self, z):
        """[summary]

        Arguments:
            z ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = z

        # constraint 1: x + y <= 3
        if (fj := x + y - 3) > 0:
            return np.array([1.0, 1.0]), fj

        # constraint 2: x - y >= 1
        if (fj := -x + y + 1) > 0:
            return np.array([-1.0, 1.0]), fj


class MyOracle:
    def assess_optim(self, z, t: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            t (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        # cut = my_oracle2(z)
        if cut := MyOracle2().assess_feas(z):
            return cut, None
        x, y = z
        # objective: maximize x + y
        f0 = x + y
        if (fj := t - f0) < 0.0:
            fj = 0.0
            t = f0
            return (-1.0 * np.array([1.0, 1.0]), fj), t
        return (-1.0 * np.array([1.0, 1.0]), fj), None


def test_case_feasible():
    """[summary]"""
    x0 = np.array([0.0, 0.0])  # initial x0
    E = ell_stable(10.0, x0)
    P = MyOracle()
    x, _, _, _ = cutting_plane_optim(P, E, float("-inf"))
    assert x is not None
    # fmt = '{:f} {} {} {}'
    # print(fmt.format(fb, niter, feasible, status))
    # print(xb)


def test_case_infeasible1():
    """[summary]"""
    x0 = np.array([100.0, 100.0])  # wrong initial guess,
    E = ell_stable(10.0, x0)  # or ellipsoid is too small
    P = MyOracle()
    x, _, _, _ = cutting_plane_optim(P, E, float("-inf"))
    assert x is None


def test_case_infeasible2():
    """[summary]"""
    x0 = np.array([0.0, 0.0])  # initial x0
    E = ell_stable(10.0, x0)
    P = MyOracle()
    x, _, _, _ = cutting_plane_optim(P, E, 100)  # wrong init best-so-far
    assert x is None
