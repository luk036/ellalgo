# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell_stable import EllStable
from ellalgo.ell_typing import OracleFeas, OracleOptim


class MyOracle2(OracleFeas):
    idx = 0
        
    def assess_feas(self, z):
        """[summary]

        Arguments:
            z ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = z

        # constraint 1: x + y <= 3
        def fn1():
            return x + y - 3

        # constraint 2: x - y >= 1
        def fn2():
            return -x + y + 1

        def grad1():
            return np.array([1.0, 1.0])

        def grad2():
            return np.array([-1.0, 1.0])

        fns = (fn1, fn2)
        grads = (grad1, grad2)

        for _ in [0, 1]:
            self.idx += 1
            if self.idx == 2:
                self.idx = 0  # round robin
            if (fj := fns[self.idx]()) > 0:
                return grads[self.idx](), fj

        return None


class MyOracle(OracleOptim):
    helper = MyOracle2()

    def assess_optim(self, z, gamma: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            gamma (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        # cut = my_oracle2(z)
        if cut := self.helper.assess_feas(z):
            return cut, None
        x, y = z
        # objective: maximize x + y
        f0 = x + y
        if (fj := gamma - f0) < 0.0:
            fj = 0.0
            gamma = f0
            return (-1.0 * np.array([1.0, 1.0]), fj), gamma
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
