"""
Test LMI
"""

from typing import Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleOptim, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_stable import EllStable
from ellalgo.oracles.lmi_old_oracle import LMIOldOracle
from ellalgo.oracles.lmi_oracle import LMIOracle

Cut = Tuple[np.ndarray, float]


class MyOracle(OracleOptim):
    idx = -1

    def __init__(self, oracle):
        """
        The function initializes arrays and matrices using numpy and assigns them to variables in the class
        instance.

        :param oracle: The `oracle` parameter in the `__init__` method is a function that takes two
            arguments `F` and `B` and returns some value. In this code snippet, the `oracle` function is being
            used to create two instances `lmi1` and `lmi2`
        """
        self.c = np.array([1.0, -1.0, 1.0])
        F1 = np.array(
            [
                [[-7.0, -11.0], [-11.0, 3.0]],
                [[7.0, -18.0], [-18.0, 8.0]],
                [[-2.0, -8.0], [-8.0, 1.0]],
            ]
        )
        B1 = np.array([[33.0, -9.0], [-9.0, 26.0]])
        F2 = np.array(
            [
                [[-21.0, -11.0, 0.0], [-11.0, 10.0, 8.0], [0.0, 8.0, 5.0]],
                [[0.0, 10.0, 16.0], [10.0, -10.0, -10.0], [16.0, -10.0, 3.0]],
                [[-5.0, 2.0, -17.0], [2.0, -6.0, 8.0], [-17.0, 8.0, 6.0]],
            ]
        )
        B2 = np.array([[14.0, 9.0, 40.0], [9.0, 91.0, 10.0], [40.0, 10.0, 15.0]])
        self.lmi1 = oracle(F1, B1)
        self.lmi2 = oracle(F2, B2)

    def assess_optim(self, xc: np.ndarray, gamma: float) -> Tuple[Cut, Optional[float]]:
        """
        This function assesses the optimality of a solution based on given constraints and a target value.

        :param xc: The parameter `xc` is a NumPy array representing a point in a multidimensional space. It
            is used as input to assess the optimality of a solution in a mathematical optimization context
        :type xc: np.ndarray
        :param gamma: Gamma is the best-so-far optimal value that is passed as an argument to the
            `assess_optim` method. It is a float value used in the optimization process to determine the
            optimality of a solution
        :type gamma: float
        :return: The `assess_optim` method returns a tuple containing a `Cut` object and an optional float
            value. The `Cut` object represents a cut in the optimization problem, while the float value
            represents the optimality measure.
        """
        for _ in range(3):
            self.idx = 0 if self.idx == 2 else self.idx + 1  # round robin

            if self.idx == 0:
                if cut := self.lmi1.assess_feas(xc):
                    return cut, None
            elif self.idx == 1:
                if cut := self.lmi2.assess_feas(xc):
                    return cut, None
            elif self.idx == 2:
                f0 = self.c.dot(xc)
                if (fj := f0 - gamma) > 0.0:
                    return (self.c, fj), None
        return (self.c, 0.0), f0


def run_lmi(oracle, space):
    """
    The `run_lmi` function takes an oracle and a Space object as input, initializes variables, performs
    optimization using cutting plane method, and returns the number of iterations.

    :param oracle: The `oracle` parameter in the `run_lmi` function is expected to be a type that is
        used as an argument for the `MyOracle` class. It seems like the `MyOracle` class is used to wrap the
        `oracle` parameter for some specific functionality within the `run_l
    :param Space: The `Space` parameter in the `run_lmi` function seems to be a class or function that
        takes two arguments - a float value `10.0` and an array `xinit`. It likely initializes some kind of
        space or environment based on these inputs
    :return: The function `run_lmi` returns the number of iterations (`num_iters`) after running the
        cutting plane optimization algorithm.
    """
    xinit = np.array([0.0, 0.0, 0.0])  # initial xinit
    ellip = space(10.0, xinit)
    omega = MyOracle(oracle)
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, float("inf"))
    assert xbest is not None
    return num_iters


def test_lmi_lazy():
    """
    The function `test_lmi_lazy` runs a specific function and asserts that the result is equal to 281.
    """
    result = run_lmi(LMIOracle, Ell)
    assert result == 281


def test_lmi_old():
    """
    The function `test_lmi_old` runs a test using an old oracle and asserts the result to be 281.
    """
    result = run_lmi(LMIOldOracle, Ell)
    assert result == 281


def test_lmi_lazy_stable():
    """
    The function `test_lmi_lazy_stable` runs a specific test and asserts the result to be equal to 281.
    """
    result = run_lmi(LMIOracle, EllStable)
    assert result == 281


def test_lmi_old_stable():
    """
    The function `test_lmi_old_stable` runs a specific test and asserts the result to be equal to 281.
    """
    result = run_lmi(LMIOldOracle, EllStable)
    assert result == 281
