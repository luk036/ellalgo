"""
Test Example 2
"""

from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_feas
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas


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


def test_case_feasible():
    """
    The function `test_case_feasible` tests the feasibility of a solution using cutting plane method.
    """
    xinit = np.array([0.0, 0.0])  # initial guess
    ellip = Ell(10.0, xinit)
    omega = MyOracle2()
    xfeas, num_iters = cutting_plane_feas(omega, ellip)
    assert xfeas is not None
    assert num_iters == 1


def test_case_infeasible():
    """
    The function `test_case_infeasible` tests the behavior of a cutting-plane algorithm with an
    infeasible initial guess.
    """
    xinit = np.array([100.0, 100.0])  # wrong initial guess
    ellip = Ell(10.0, xinit)
    omega = MyOracle2()
    xfeas, num_iters = cutting_plane_feas(omega, ellip)
    assert xfeas is None
    assert num_iters == 1
