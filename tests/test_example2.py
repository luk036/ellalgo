# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_feas, OracleFeas
from ellalgo.ell import Ell


class MyOracle(OracleFeas):
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


def test_case_feasible():
    """[summary]"""
    xinit = np.array([0.0, 0.0])  # initial guess
    ellip = Ell(10.0, xinit)
    omega = MyOracle()
    x_feas, num_iters = cutting_plane_feas(omega, ellip)
    assert x_feas is not None
    print(num_iters)


def test_case_infeasible():
    """[summary]"""
    xinit = np.array([100.0, 100.0])  # wrong initial guess
    ellip = Ell(10.0, xinit)
    omega = MyOracle()
    x_feas, num_iters = cutting_plane_feas(omega, ellip)
    assert x_feas is None
    print(num_iters)
