"""
Test Quasiconvex (with Round Robin)
"""

from __future__ import print_function
import math
from pytest import approx
import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim, Options
from ellalgo.ell import Ell
from ellalgo.ell_stable import EllStable
from ellalgo.ell_typing import OracleOptim


class MyQuasicvxOracle(OracleOptim):
    idx: int = -1  # for round robin
    y: float
    tmp3: float

    def __init__(self):
        """
        The function initializes two tuples containing function and gradient references.
        """
        self.fns = (self.fn1, self.fn2)
        self.grads = (self.grad1, self.grad2)

    def fn1(self, sqrtx, logy, _):
        """
        The function calculates the difference between the square of a given value and another value.

        :param sqrtx: The parameter `sqrtx` represents the square root of a value
        :param logy: The parameter `logy` represents the upper limit for the square of the square root of `x`
        :param _: The underscore symbol (_) is commonly used as a placeholder variable in Python to indicate
            that the value is not going to be used in the function. In this context, it seems that the third
            parameter is not used in the function `fn1`
        :return: The function `fn1` is returning the value of `sqrtx * sqrtx - logy`.
        """
        return sqrtx * sqrtx - logy

    def fn2(self, sqrtx, logy, gamma):
        """
        The function calculates the value of -sqrt(x) plus gamma times the exponential of y.

        :param sqrtx: The `sqrtx` parameter represents the square root of a value
        :param logy: The parameter `logy` appears to represent the natural logarithm of `y`
        :param gamma: Gamma is a constant value used in the calculation within the function
        :return: The function `fn2` is returning the value of `-sqrtx + self.tmp3`.
        """
        self.y = math.exp(logy)
        self.tmp3 = gamma * self.y
        return -sqrtx + self.tmp3

    def grad1(self, sqrtx):
        """
        The function `grad1` calculates the gradient of a function with respect to the input `sqrtx`.

        :param sqrtx: The `sqrtx` parameter in the `grad1` function seems to represent the square root of a
            variable `x`. The function calculates the gradient of a function with respect to `sqrtx` and returns
            a numpy array with two elements: `2 * sqrtx` and `-1.0
        :return: The function `grad1` is returning a NumPy array with two elements. The first element is `2
            * sqrtx` and the second element is `-1.0`.
        """
        return np.array([2 * sqrtx, -1.0])

    def grad2(self, _):
        """
        The `grad2` function returns a NumPy array with values -1.0 and the value of `self.tmp3`.

        :param _: The parameter "_" is typically used as a placeholder when the value is not needed or not
            relevant in the context of the function
        :return: The `grad2` function is returning a NumPy array with two elements: -1.0 and the value of
            `self.tmp3`.
        """
        return np.array([-1.0, self.tmp3])

    def assess_optim(self, xc, gamma: float):
        """
        The function assess_optim takes input parameters xc and gamma, iterates through a loop, and returns a
        tuple based on certain conditions.

        :param xc: The `xc` parameter in the `assess_optim` method seems to be a tuple containing two
            elements: `sqrtx` and `logy`. These values are then used in the method for further calculations
        :param gamma: Gamma is a float value representing the best-so-far optimal value. It is used in the
            `assess_optim` method for calculations and comparisons
        :type gamma: float
        :return: The function `assess_optim` returns a tuple containing two elements. The first element is a
            tuple containing an array and a float value, and the second element is either `None` or a float
            value.
        """
        sqrtx, logy = xc

        for _ in [0, 1]:
            self.idx += 1
            if self.idx == 2:
                self.idx = 0  # round robin
            if (fj := self.fns[self.idx](sqrtx, logy, gamma)) > 0:
                return (self.grads[self.idx](sqrtx), fj), None

        gamma = sqrtx / self.y
        return (np.array([-1.0, sqrtx]), 0), gamma


def test_case_feasible():
    """
    The function `test_case_feasible` sets up a test case for optimization using cutting plane method
    and asserts the expected results.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    options = Options()
    options.tolerance = 1e-8
    omega = MyQuasicvxOracle()
    xbest, fbest, niters = cutting_plane_optim(omega, ellip, 0.0, options)
    assert xbest is not None
    assert niters == 35
    assert fbest == approx(0.4288687202295896)
    assert xbest[0] * xbest[0] == approx(0.4965439352179321)
    assert math.exp(xbest[1]) == approx(1.6430639574974657)


def test_case_infeasible1():
    """
    The function `test_case_infeasible1` tests for infeasibility in an optimization problem.
    """
    xinit = np.array([100.0, 100.0])  # wrong initial guess,
    ellip = Ell(10.0, xinit)  # or ellipsoid is too small
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is None


def test_case_infeasible2():
    """
    The function `test_case_infeasible2` initializes variables and asserts that the result is None.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 100)  # wrong init best-so-far
    assert xbest is None


def test_case_feasible_stable():
    """
    The function `test_case_feasible_stable` tests the optimization of a convex function using cutting
    plane method with specific assertions.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = EllStable(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, fbest, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    assert fbest == approx(0.4288819424771711)
    assert xbest[0] * xbest[0] == approx(0.5000000285975479)
    assert math.exp(xbest[1]) == approx(1.6487213178704083)


def test_case_infeasible1_stable():
    """
    The function `test_case_infeasible1_stable` tests for infeasibility by using a wrong initial guess
    or an ellipsoid that is too small.
    """
    xinit = np.array([100.0, 100.0])  # wrong initial guess,
    ellip = EllStable(10.0, xinit)  # or ellipsoid is too small
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is None


def test_case_infeasible2_stable():
    """
    The function `test_case_infeasible2_stable` initializes variables and performs optimization using
    cutting plane method, asserting that the result is `None`.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = EllStable(10.0, xinit)
    omega = MyQuasicvxOracle()
    xbest, _, _ = cutting_plane_optim(omega, ellip, 100)  # wrong init best-so-far
    assert xbest is None
