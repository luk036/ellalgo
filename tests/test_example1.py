"""
Test Example 1 (with round robin)
"""

from __future__ import print_function

import numpy as np
from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleOptim

num_constraints = 3


class MyOracle1(OracleOptim):
    """
    This Python class `MyOracle1` contains a method `assess_optim` that assesses optimization based on
    given parameters and returns specific values accordingly.
    """

    idx = -1  # for round robin

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

        for _ in range(num_constraints):
            self.idx += 1
            if self.idx == num_constraints:
                self.idx = 0  # round robin

            if self.idx == 0:
                if (fj := f0 - 3.0) > 0.0:
                    return ((np.array([1.0, 1.0]), fj), None)
            elif self.idx == 1:
                if (fj := -x + y + 1.0) > 0.0:
                    return ((np.array([-1.0, 1.0]), fj), None)
            elif self.idx == 2:
                if (fj := gamma - f0) > 0.0:
                    return ((np.array([-1.0, -1.0]), fj), None)
            else:
                raise ValueError("Unexpected index value")

        return ((np.array([-1.0, -1.0]), 0.0), f0)


def test_case_feasible():
    """
    The function `test_case_feasible` sets up a test case for a cutting plane optimization algorithm
    with specific parameters and asserts the expected outcome.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    options = Options()
    options.tolerance = 1e-10
    omega = MyOracle1()
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, float("-inf"), options)
    assert xbest is not None
    assert num_iters == 25


def test_case_infeasible1():
    """
    The function `test_case_infeasible1` tests for infeasibility by providing a wrong initial guess or
    an ellipsoid that is too small.
    """
    xinit = np.array([100.0, 100.0])  # wrong initial guess,
    ellip = Ell(10.0, xinit)  # or ellipsoid is too small
    omega = MyOracle1()
    xbest, _, _ = cutting_plane_optim(omega, ellip, float("-inf"))
    assert xbest is None


def test_case_infeasible2():
    """
    The function `test_case_infeasible2` initializes variables and asserts that the best solution is
    None.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    omega = MyOracle1()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 100)  # wrong init best-so-far
    assert xbest is None
