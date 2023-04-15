# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_feas
from ellalgo.ell import Ell


class MyOracle:
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
    ell_info = cutting_plane_feas(omega, ellip)
    assert ell_info.feasible
    print(ell_info.num_iters)


def test_case_infeasible():
    """[summary]"""
    xinit = np.array([100.0, 100.0])  # wrong initial guess
    ellip = Ell(10.0, xinit)
    omega = MyOracle()
    ell_info = cutting_plane_feas(omega, ellip)
    assert ell_info.num_iters == 0  # small
    assert not ell_info.feasible
