# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas, OracleOptim


class MyOracleFeas(OracleFeas):
    idx = 0
    gamma = 0.0

    # constraint 1: x + y <= 3
    def fn1(self, x, y):
        return x + y - 3

    # constraint 2: x - y >= 1
    def fn2(self, x, y):
        return -x + y + 1

    def fn0(self, x, y):
        return self.gamma - (x + y)

    def grad1(self):
        return np.array([1.0, 1.0])

    def grad2(self):
        return np.array([-1.0, 1.0])

    def grad0(self):
        return np.array([-1.0, -1.0])

    def __init__(self):
        self.fns = (self.fn1, self.fn2, self.fn0)
        self.grads = (self.grad1, self.grad2, self.grad0)

    def assess_feas(self, z):
        """[summary]

        Arguments:
            z ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = z

        for _ in range(3):
            self.idx = (self.idx + 1) % 3  # round robin
            if (fj := self.fns[self.idx](x, y)) > 0:
                return self.grads[self.idx](), fj
        return None


class MyOracle(OracleOptim):
    helper = MyOracleFeas()

    def assess_optim(self, z, gamma: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            gamma (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        self.helper.gamma = gamma
        if cut := self.helper.assess_feas(z):
            return cut, None
        x, y = z
        # objective: maximize x + y
        return (np.array([-1.0, -1.0]), 0.0), x + y


class MyOracleFeas2(OracleFeas):
    gamma = 0.0

    # constraint 1: x + y <= 3
    def fn1(self, x, y):
        return x + y - 3

    # constraint 2: x - y >= 1
    def fn2(self, x, y):
        return -x + y + 1

    def fn0(self, x, y):
        return self.gamma - (x + y)

    def grad1(self):
        return np.array([1.0, 1.0])

    def grad2(self):
        return np.array([-1.0, 1.0])

    def grad0(self):
        return np.array([-1.0, -1.0])

    def __init__(self):
        self.fns = (self.fn1, self.fn2, self.fn0)
        self.grads = (self.grad1, self.grad2, self.grad0)

    def assess_feas(self, z):
        """[summary]

        Arguments:
            z ([type]): [description]

        Returns:
            [type]: [description]
        """
        x, y = z
        for idx in range(3):
            if (fj := self.fns[idx](x, y)) > 0:
                return self.grads[idx](), fj
        return None


class MyOracle2(OracleOptim):
    helper = MyOracleFeas2()

    def assess_optim(self, z, gamma: float):
        """[summary]

        Arguments:
            z ([type]): [description]
            gamma (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        self.helper.gamma = gamma
        if cut := self.helper.assess_feas(z):
            return cut, None
        x, y = z
        # objective: maximize x + y
        return (np.array([-1.0, -1.0]), 0.0), x + y


def run_example1(omega):
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    options = Options()
    options.tolerance = 1e-10
    xbest, _, num_iters = cutting_plane_optim(omega(), ellip, float("-inf"), options)
    assert xbest is not None
    return num_iters


def test_bm_with_round_robin(benchmark):
    num_iters = benchmark(run_example1, MyOracle)
    assert num_iters == 25


def test_bm_without_round_robin(benchmark):
    num_iters = benchmark(run_example1, MyOracle2)
    assert num_iters == 25
