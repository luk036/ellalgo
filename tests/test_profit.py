"""
Test Maximum Profit
"""

from __future__ import annotations
import numpy as np
from typing import Type

from ellalgo.cutting_plane import cutting_plane_optim, cutting_plane_optim_q
from ellalgo.ell import Ell
from ellalgo.ell_stable import EllStable
from ellalgo.ell_typing import ArrayType, SearchSpace, SearchSpaceQ
from ellalgo.oracles.profit_oracle import ProfitOracle, ProfitQOracle, ProfitRbOracle

price, amplitude, constant = 20.0, 40.0, 30.5
params = price, amplitude, constant
alpha, beta = 0.1, 0.4
value1, value2 = 10.0, 35.0
alpha_beta_array = np.array([alpha, beta])
value_array = np.array([value1, value2])
radius_array = np.array([100.0, 100.0])  # initial ellipsoid (sphere)


def run_profit(E: Type[SearchSpace]) -> int:
    """
    The function `run_profit` calculates the number of iterations needed to optimize profit using
    cutting plane optimization.

    :param E: It looks like the `run_profit` function is using several parameters, but the definition of
        the `E` function is missing. Could you please provide the definition or description of the `E`
        function so that I can better understand how it is being used in the `run_profit` function?
    :return: The function `run_profit` returns the number of iterations (`num_iters`) performed during
        the optimization process.
    """
    ellip = E(radius_array, np.array([0.0, 0.0]))
    omega = ProfitOracle(params, alpha_beta_array, value_array)
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    return num_iters


def run_profit_rb(E: Type[SearchSpace]) -> int:
    """
    The function `run_profit_rb` calculates the number of iterations required to optimize a profit
    function using cutting-plane optimization.

    :param E: It looks like the function `run_profit_rb` is missing the definition of the `E` parameter.
        In order to provide you with the correct information, could you please provide the definition or
        explanation of the `E` parameter that is required by the function `run_profit_rb`?
    :return: The function `run_profit_rb` returns the number of iterations `num_iters` after running the
        cutting plane optimization algorithm with the given parameters and constraints.
    """
    epsilon1 = 0.003
    epsilon2 = 0.007
    epsilon3 = epsilon4 = epsilon5 = 1.0
    ellip = E(radius_array, np.array([0.0, 0.0]))
    omega = ProfitRbOracle(params, alpha_beta_array, value_array, (epsilon1, epsilon2, epsilon3, epsilon4, epsilon5))
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    return num_iters


def run_profit_q(E: Type[SearchSpaceQ[ArrayType]]) -> int:
    """
    The function `run_profit_q` optimizes a profit function using a cutting plane optimization
    algorithm.

    :param E: It looks like the function `run_profit_q` is using some parameters such as `r`,
        `np.array([0.0, 0.0])`, `params`, `a`, and `v` without explicitly defining them in the function
    :return: The function `run_profit_q` returns the number of iterations (`num_iters`) performed during
        the optimization process.
    """
    ellip = EllStable(radius_array, np.array([0.0, 0.0]))
    omega = ProfitQOracle(params, alpha_beta_array, value_array)
    xbest, _, num_iters = cutting_plane_optim_q(omega, ellip, 0.0)
    assert xbest is not None
    return num_iters


def test_profit_oracle() -> None:
    epsilon1 = 0.003
    epsilon2 = 0.007
    epsilon3 = epsilon4 = epsilon5 = 1.0
    omega = ProfitRbOracle(params, alpha_beta_array, value_array, (epsilon1, epsilon2, epsilon3, epsilon4, epsilon5))
    test_point = np.array([0.0, 0.0])
    cut = omega.assess_optim(test_point, 0.0)
    assert cut is not None


def test_profit_ell() -> None:
    """
    The function `test_profit_ell` checks if the number of iterations required to run the `run_profit`
    function with the `Ell` parameter is equal to 83.
    """
    num_iters = run_profit(Ell)
    assert num_iters == 83


def test_profit_ell_stable() -> None:
    """
    The function `test_profit_ell_stable` tests the number of iterations required to run the `EllStable`
    algorithm for profit calculation.
    """
    num_iters = run_profit(EllStable)
    assert num_iters == 83


def test_profit_rb_ell() -> None:
    """
    The function `test_profit_rb_ell` tests the number of iterations required to run a profit
    calculation using a specific method.
    """
    num_iters = run_profit_rb(Ell)
    assert num_iters == 90


def test_profit_rb_ell_stable() -> None:
    """
    The function `test_profit_rb_ell_stable` tests the number of iterations required for a specific
    profit calculation algorithm to converge using the EllStable method.
    """
    num_iters = run_profit_rb(EllStable)
    assert num_iters == 90


def test_profit_q_ell() -> None:
    """
    The function `test_profit_q_ell` tests the number of iterations required to run a profit calculation
    using a specific parameter.
    """
    num_iters = run_profit_q(EllStable)
    assert num_iters == 29


def test_profit_q_ell_stable() -> None:
    """
    The function `test_profit_q_ell_stable` tests the number of iterations required to run a profit
    calculation using the `EllStable` algorithm.
    """
    num_iters = run_profit_q(EllStable)
    assert num_iters == 29
