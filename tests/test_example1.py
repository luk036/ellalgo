# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell_stable import EllStable


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
    def assess_optim(self, z, tea: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            tea (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        # cut = my_oracle2(z)
        if cut := MyOracle2().assess_feas(z):
            return cut, None
        x, y = z
        # objective: maximize x + y
        f0 = x + y
        if (fj := tea - f0) < 0.0:
            fj = 0.0
            tea = f0
            return (-1.0 * np.array([1.0, 1.0]), fj), tea
        return (-1.0 * np.array([1.0, 1.0]), fj), None


def test_case_feasible():
    """[summary]"""
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = EllStable(10.0, xinit)
    omega = MyOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, float("-inf"))
    assert xbest is not None
    # fmt = '{:f} {} {} {}'
    # print(fmt.format(fbest, niter, feasible, status))
    # print(xbest)


def test_case_infeasible1():
    """[summary]"""
    xinit = np.array([100.0, 100.0])  # wrong initial guess,
    ellip = EllStable(10.0, xinit)  # or ellipsoid is too small
    omega = MyOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, float("-inf"))
    assert xbest is None


def test_case_infeasible2():
    """[summary]"""
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = EllStable(10.0, xinit)
    omega = MyOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 100)  # wrong init best-so-far
    assert xbest is None
