"""
Test Example 1
"""

from __future__ import print_function

import numpy as np
from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas, OracleOptim


class MyOracle2(OracleFeas):
    """
    The `MyOracle2` class in Python defines functions for calculating mathematical expressions and
    gradients, with a method to assess feasibility based on function values.
    """

    idx = 0

    def fn1(self, x, y):
        """
        The function `fn1` calculates the difference between the sum of two input values `x` and `y` and the
        value 3.

        :param x: The parameter x represents the value of the first variable in the function
        :param y: The parameter `y` represents the value of the variable `y` in the mathematical expression
        `x + y <= 3`. It is used in the context of the constraint that the sum of `x` and `y` should be less
        than or equal to 3 in the function `fn
        :return: The function `fn1` is returning the result of the expression `x + y - 3`.
        """
        return x + y - 3

    def fn2(self, x, y):
        """
        The function fn2 calculates the value of -x + y + 1.

        :param x: The parameter x represents a numerical value in the function
        :param y: The parameter `y` is a variable that is used in the function `fn2` to calculate the
        expression `-x + y + 1`
        :return: The function `fn2` is returning the value of `-x + y + 1`.
        """
        return -x + y + 1

    def grad1(self):
        """
        The function `grad1` returns a NumPy array with two elements, both set to 1.0.
        :return: The function `grad1` is returning a NumPy array with the values `[1.0, 1.0]`.
        """
        return np.array([1.0, 1.0])

    def grad2(self):
        """
        The `grad2` function returns a NumPy array with two elements, [-1.0, 1.0].
        :return: The function `grad2` is returning a NumPy array `[-1.0, 1.0]`.
        """
        return np.array([-1.0, 1.0])

    def __init__(self):
        """
        The function initializes two tuples containing function and gradient references.
        """
        self.fns = (self.fn1, self.fn2)
        self.grads = (self.grad1, self.grad2)

    def assess_feas(self, xc):
        """
        The `assess_feas` function iterates through a list of functions and returns the result of the first
        function that returns a positive value along with the corresponding gradient.

        :param xc: The `assess_feas` method takes a tuple `xc` as input, which is then unpacked into variables
        `x` and `y`. The method then iterates over a list `[0, 1]` and performs some operations based on the
        values of `x` and `
        :return: The `assess_feas` method returns a tuple containing the result of calling the function at
        index `self.idx` from the `grads` list and the value of `fj` if it is greater than 0. If none of the
        conditions are met, it returns `None`.
        """
        x, y = xc

        for _ in [0, 1]:
            self.idx += 1
            if self.idx == 2:
                self.idx = 0  # round robin
            if (fj := self.fns[self.idx](x, y)) > 0:
                return self.grads[self.idx](), fj
        return None


class MyOracle1(OracleOptim):
    """
    This Python class `MyOracle1` contains a method `assess_optim` that assesses optimization based on
    given parameters and returns specific values accordingly.
    """

    helper = MyOracle2()

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
        # cut = my_oracle2(z)
        if cut := self.helper.assess_feas(xc):
            return cut, None
        x, y = xc
        # objective: maximize x + y
        f0 = x + y
        if (fj := gamma - f0) > 0.0:
            return (-1.0 * np.array([1.0, 1.0]), fj), None

        gamma = f0
        return (-1.0 * np.array([1.0, 1.0]), 0.0), gamma


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
