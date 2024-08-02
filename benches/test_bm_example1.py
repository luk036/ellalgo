# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleOptim


class MyOracle1(OracleOptim):
    """
    This Python class `MyOracle1` contains a method `assess_optim` that assesses optimization based on
    given parameters and returns specific values accordingly.
    """

    def __init__(self):
        """
        Creates a new `MyOracle` instance with the `idx` field initialized to 0.

        This is the constructor for the `MyOracle` class, which is the main entry point for
        creating new instances of this type. It initializes the `idx` field to 0, which is the
        default value for this field.

        Examples:
        >>> oracle = MyOracle()
        >>> assert oracle.idx == 0
        """
        self.idx = -1

    def assess_optim(self, xc, gamma: float):
        """
        The function assess_optim assesses feasibility and optimality of a given point based on a specified
        gamma value.

        :param xc: The `xc` parameter in the `assess_optim` method appears to represent a point in a
            two-dimensional space, as it is being unpacked into `x` and `y` coordinates
        :param gamma: Gamma is a parameter used in the `assess_optim` method. It is a float value that is
            compared with the sum of `x` and `y` in the objective function. The method returns different values
            based on the comparison of `gamma` with the sum of `x` and
        :type gamma: float
        :return: The `assess_optim` method returns a tuple containing two elements. The first element is a
            tuple containing an array `[-1.0, -1.0]` and either the value of `fj` (if `fj > 0.0`) or `0.0` (if
            `fj <= 0.0`). The second element of the tuple is
        """
        x, y = xc
        f0 = x + y

        for _ in range(3):
            self.idx = (self.idx + 1) % 3  # round robin

            if self.idx == 0:
                fj = f0 - 3.0
            elif self.idx == 1:
                fj = -x + y + 1.0
            elif self.idx == 2:
                fj = gamma - f0
            else:
                raise ValueError("Unexpected index value")

            if fj > 0.0:
                if self.idx == 0:
                    return ((np.array([1.0, 1.0]), fj), None)
                elif self.idx == 1:
                    return ((np.array([-1.0, 1.0]), fj), None)
                elif self.idx == 2:
                    return ((np.array([-1.0, -1.0]), fj), None)

        gamma = f0
        return ((np.array([-1.0, -1.0]), 0.0), gamma)


class MyOracle2(OracleOptim):
    """
    This Python class `MyOracle2` contains a method `assess_optim` that assesses optimization based on
    given parameters and returns specific values accordingly.
    """

    def assess_optim(self, xc, gamma: float):
        """
        The function assess_optim assesses feasibility and optimality of a given point based on a specified
        gamma value.

        :param xc: The `xc` parameter in the `assess_optim` method appears to represent a point in a
            two-dimensional space, as it is being unpacked into `x` and `y` coordinates
        :param gamma: Gamma is a parameter used in the `assess_optim` method. It is a float value that is
            compared with the sum of `x` and `y` in the objective function. The method returns different values
            based on the comparison of `gamma` with the sum of `x` and
        :type gamma: float
        :return: The `assess_optim` method returns a tuple containing two elements. The first element is a
            tuple containing an array `[-1.0, -1.0]` and either the value of `fj` (if `fj > 0.0`) or `0.0` (if
            `fj <= 0.0`). The second element of the tuple is
        """
        x, y = xc
        f0 = x + y
        if (fj := f0 - 3.0) > 0.0:
            return ((np.array([1.0, 1.0]), fj), None)
        if (fj := -x + y + 1.0) > 0.0:
            return ((np.array([-1.0, 1.0]), fj), None)
        if (fj := gamma - f0) > 0.0:
            return ((np.array([-1.0, -1.0]), fj), None)
        return ((np.array([-1.0, -1.0]), 0.0), f0)


def run_example1(omega):
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    options = Options()
    options.tolerance = 1e-10
    xbest, _, num_iters = cutting_plane_optim(omega(), ellip, float("-inf"), options)
    assert xbest is not None
    return num_iters


def test_bm_with_round_robin(benchmark):
    num_iters = benchmark(run_example1, MyOracle1)
    assert num_iters == 25


def test_bm_without_round_robin(benchmark):
    num_iters = benchmark(run_example1, MyOracle2)
    assert num_iters == 25
