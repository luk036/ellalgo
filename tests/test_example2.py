# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_feas
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas


class MyOracle(OracleFeas):
    idx = 0

    # constraint 1: x + y <= 3
    def fn1(self, x, y):
        return x + y - 3

    # constraint 2: x - y >= 1
    def fn2(self, x, y):
        return -x + y + 1

    def grad1(self):
        return np.array([1.0, 1.0])

    def grad2(self):
        return np.array([-1.0, 1.0])

    def __init__(self):
        self.fns = (self.fn1, self.fn2)
        self.grads = (self.grad1, self.grad2)

    def assess_feas(self, z):
        """[summary]

        Arguments:
            z ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = z

        for _ in [0, 1]:
            self.idx += 1
            if self.idx == 2:
                self.idx = 0  # round robin
            if (fj := self.fns[self.idx](x, y)) > 0:
                return self.grads[self.idx](), fj
        return None


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
