"""
Test Cutting Plane
"""

from __future__ import print_function

import numpy as np
import pytest
from ellalgo.cutting_plane import (
    Options,
    cutting_plane_feas,
    cutting_plane_optim,
    cutting_plane_optim_q,
    bsearch,
)
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas, OracleOptim, OracleOptimQ, OracleBS
from typing import Tuple, Optional


class MyOracleFeas(OracleFeas):
    """
    The `MyOracleFeas` class in Python defines a method `assess_feas` that assesses the feasibility of a
    given point `xc` based on a specific constraint.
    """

    def assess_feas(self, xc) -> Optional[Tuple[np.ndarray, float]]:
        """
        The function `assess_feas` assesses the feasibility of a given point `xc` and returns a tuple
        containing a numpy array and a float value if a certain condition is met, otherwise it returns
        None.

        :param xc: The parameter `xc` seems to be a list or tuple of two values, where the first value is
        assigned to `x` and the second value is assigned to `y`
        :return: The `assess_feas` method returns either a tuple `(np.array([1.0, 1.0]), fj)` if the
        condition `(fj := xc[0] + xc[1] - 3.0) > 0.0` is true, or `None` if the condition is false.
        """
        x, y = xc
        if (fj := x + y - 3.0) > 0.0:
            return (np.array([1.0, 1.0]), fj)
        return None


class MyOracleInfeas(OracleFeas):
    """
    The `MyOracleInfeas` class contains a method to assess feasibility based on a given input, always
    returning a specific tuple.
    """

    def assess_feas(self, xc) -> Optional[Tuple[np.ndarray, float]]:
        """
        The function `assess_feas` takes a parameter `xc` and always returns a tuple containing a numpy
        array `[1.0, 1.0]` and a float `1.0`.

        :param xc: The parameter `xc` is a variable that is passed to the `assess_feas` function. It is
        not used within the function in this specific implementation, as the function always returns a
        fixed tuple `(np.array([1.0, 1.0]), 1.0)`
        :return: A tuple containing a NumPy array `[1.0, 1.0]` and the float value `1.0`.
        """
        return (np.array([1.0, 1.0]), 1.0)


class MyOracleOptim(OracleOptim):
    """
    The `MyOracleOptim` class implements an optimization oracle with a specific assessment logic based on
    the input variables `xc` and `gamma`.
    """

    def assess_optim(
        self, xc, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], Optional[float]]:
        """
        The `assess_optim` function in Python assesses an optimization problem by calculating a value `f0`
        based on input `xc`, and then checking against constraints `f1` and `f2` to determine the outcome.

        :param xc: The `xc` parameter in the `assess_optim` method seems to be a tuple or list with two
        elements, which are unpacked into `x` and `y` variables
        :param gamma: The `gamma` parameter in the `assess_optim` function is a float value that is used
        for comparison with `f0`
        :type gamma: float
        :return: The `assess_optim` method returns a tuple. The first element of the tuple is another
        tuple containing a numpy array and a float value. The second element of the tuple is an optional
        float value.
        """
        x, y = xc
        f0 = x + y
        if (f1 := x - 1.0) > 0.0:
            return ((np.array([1.0, 0.0]), f1), None)
        if (f2 := y - 1.0) > 0.0:
            return ((np.array([0.0, 1.0]), f2), None)
        if (f3 := f0 - gamma) < 0.0:
            return ((np.array([-1.0, -1.0]), -f3), None)
        return ((np.array([-1.0, -1.0]), 0.0), f0)


class MyOracleOptimQ(OracleOptimQ):
    """
    The `MyOracleOptimQ` class implements an optimization oracle for a specific problem, assessing the
    optimality of a given solution `xc` with respect to a threshold `gamma`.
    """

    def assess_optim_q(self, xc, gamma: float, retry: bool):
        """
        The `assess_optim_q` function assesses an optimization problem with quadratic constraints and
        returns a tuple with cut information, a solution vector, an updated gamma value, and a boolean
        indicating if an alternative solution exists.

        :param xc: The `xc` parameter seems to be a list or array of values, where `x` is the first
        element and `y` is the second element
        :param gamma: The `gamma` parameter is a float that represents the current best-so-far objective
        value
        :type gamma: float
        :param retry: The `retry` parameter is a boolean flag that indicates whether the assessment is a
        retry attempt. If `retry` is `True`, it means the previous attempt failed and the oracle should
        try to find an alternative solution
        :type retry: bool
        :return: The `assess_optim_q` method returns a tuple containing the following elements:
        1. A tuple `(g, fj)` where `g` is a numpy array and `fj` is a float.
        2. A numpy array `x_q` which is the quantized solution.
        3. A float `gamma` which is the updated best-so-far objective value.
        4. A boolean value indicating if there are more alternative solutions.
        """
        x, y = xc
        f0 = x + y
        if (f1 := x - 1.0) > 0.0:
            return ((np.array([1.0, 0.0]), f1), None, None, True)
        if (f2 := y - 1.0) > 0.0:
            return ((np.array([0.0, 1.0]), f2), None, None, True)
        if (f3 := f0 - gamma) < 0.0:
            return ((np.array([-1.0, -1.0]), -f3), None, None, True)

        x_q = np.round(xc)
        if (f1 := x_q[0] - 1.0) > 0.0:
            return ((np.array([1.0, 0.0]), f1), x_q, None, not retry)
        if (f2 := x_q[1] - 1.0) > 0.0:
            return ((np.array([0.0, 1.0]), f2), x_q, None, not retry)
        if (f3 := x_q[0] + x_q[1] - gamma) < 0.0:
            return ((np.array([-1.0, -1.0]), -f3), x_q, None, not retry)
        return ((np.array([-1.0, -1.0]), 0.0), x_q, f0, not retry)


class MyOracleBS(OracleBS):
    """
    The `MyOracleBS` class in Python defines a method `assess_bs` that always returns `False`.
    """

    def assess_bs(self, gamma: float) -> bool:
        """
        The function `assess_bs` takes a float `gamma` as input and always returns `False`.

        :param gamma: The parameter `gamma` is a float, but it is not used in the function's logic
        :type gamma: float
        :return: The `assess_bs` method is returning a boolean value, which is `False`.
        """
        return gamma > 0


def test_cutting_plane_feas():
    """
    The function `test_cutting_plane_feas` tests the feasibility of a cutting plane algorithm by
    initializing a search space and an oracle, and then asserting that a feasible solution is found
    within a certain number of iterations.
    """
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleFeas()
    options = Options()
    options.max_iters = 200
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is not None
    assert num_iters == 0


def test_cutting_plane_feas_no_soln():
    """
    The function `test_cutting_plane_feas_no_soln` tests a scenario where no feasible solution is
    expected to be found by the cutting plane algorithm.
    """
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleInfeas()
    options = Options()
    options.max_iters = 200
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is None
    assert num_iters == 2


def test_cutting_plane_optim():
    """
    The function `test_cutting_plane_optim` tests an optimization algorithm with a specific oracle and
    initial conditions, asserting that the results meet the expected values.
    """
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim()
    options = Options()
    options.max_iters = 200
    xbest, fbest, num_iters = cutting_plane_optim(omega, ellip, 0.0, options)
    assert xbest is not None
    assert fbest == pytest.approx(2.0)
    assert num_iters == 145


def test_cutting_plane_optim_no_soln():
    """
    The function `test_cutting_plane_optim_no_soln` tests a scenario where no optimal solution is
    found within the specified number of iterations.
    """
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim()
    options = Options()
    options.max_iters = 4
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 100.0, options)
    assert xbest is None
    assert num_iters == 0


def test_cutting_plane_optim_q():
    """
    The function `test_cutting_plane_optim_q` tests a quantized cutting plane optimization algorithm.
    """
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptimQ()
    options = Options()
    options.max_iters = 200
    xbest, fbest, num_iters = cutting_plane_optim_q(omega, ellip, 0.0, options)
    assert xbest is not None
    assert fbest == pytest.approx(2.0)
    assert num_iters == 145


def test_cutting_plane_optim_q_no_soln():
    """
    The function `test_cutting_plane_optim_q_no_soln` tests a scenario where no solution is found for a
    quantized optimization problem.
    """
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptimQ()
    options = Options()
    options.max_iters = 20
    xbest, _, num_iters = cutting_plane_optim_q(omega, ellip, 100.0, options)
    assert xbest is None
    assert num_iters == 0


def test_bsearch():
    """
    The `test_bsearch` function tests the `bsearch` function with a specific oracle and asserts that the
    result is close to 0.0 with a tolerance of 1e-6.
    """
    omega = MyOracleBS()
    options = Options()
    options.tolerance = 1e-7
    gamma, num_iters = bsearch(omega, (-100.0, 100.0), options)
    assert gamma > 0.0
    assert gamma < 2e-7
    assert num_iters == 30


def test_bsearch_no_soln():
    """
    The function `test_bsearch_no_soln` tests the behavior of the `bsearch` function when no solution
    is expected.
    """
    omega = MyOracleBS()
    options = Options()
    options.max_iters = 20
    gamma, num_iters = bsearch(omega, (-100.0, -50.0), options)
    assert gamma == -50.0
    assert num_iters == 20
