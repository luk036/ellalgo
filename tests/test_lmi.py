from typing import Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleOptim, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.ell_stable import EllStable
from ellalgo.oracles.lmi_old_oracle import LMIOldOracle
from ellalgo.oracles.lmi_oracle import LMIOracle

Cut = Tuple[np.ndarray, float]


class MyOracle(OracleOptim):
    def __init__(self, oracle):
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
        """[summary]

        Arguments:
            xc (np.ndarray): [description]
            gamma (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: [description]
        """
        if cut := self.lmi1.assess_feas(xc):
            return cut, None

        if cut := self.lmi2.assess_feas(xc):
            return cut, None

        f0 = self.c.dot(xc)
        if (fj := f0 - gamma) > 0.0:
            return (self.c, fj), None
        return (self.c, 0.0), f0


def run_lmi(oracle, Space):
    """[summary]

    Arguments:
        oracle ([type]): [description]

    Keyword Arguments:
        duration (float): [description] (default: {0.000001})

    Returns:
        [type]: [description]
    """
    xinit = np.array([0.0, 0.0, 0.0])  # initial xinit
    ellip = Space(10.0, xinit)
    omega = MyOracle(oracle)
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, float("inf"))
    assert xbest is not None
    return num_iters


def test_lmi_lazy():
    """[summary]

    Arguments:
         ([type]): [description]
    """
    result = run_lmi(LMIOracle, Ell)
    assert result == 281


def test_lmi_old():
    """[summary]

    Arguments:
         ([type]): [description]
    """
    result = run_lmi(LMIOldOracle, Ell)
    assert result == 281


def test_lmi_lazy_stable():
    """[summary]

    Arguments:
         ([type]): [description]
    """
    result = run_lmi(LMIOracle, EllStable)
    assert result == 281


def test_lmi_old_stable():
    """[summary]

    Arguments:
         ([type]): [description]
    """
    result = run_lmi(LMIOldOracle, EllStable)
    assert result == 281
