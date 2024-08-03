"""
Test Example 2
"""

from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_feas
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas

num_constraints = 2


class MyOracle2(OracleFeas):
    """
    The `MyOracle2` class in Python defines functions for calculating mathematical expressions and
    gradients, with a method to assess feasibility based on function values.
    """

    idx = -1  # for round robin

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
        for _ in range(num_constraints):
            self.idx += 1
            if self.idx == num_constraints:
                self.idx = 0  # round robin

            if self.idx == 0:
                if (fj := x + y - 3.0) > 0.0:
                    return np.array([1.0, 1.0]), fj
            elif self.idx == 1:
                if (fj := -x + y + 1.0) > 0.0:
                    return np.array([-1.0, 1.0]), fj
            else:
                raise ValueError("Unexpected index value")

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
    assert num_iters == 0
