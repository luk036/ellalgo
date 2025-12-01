"""
Test LMI0 Oracle
"""

import numpy as np

from ellalgo.oracles.lmi0_oracle import LMI0Oracle


def test_lmi0_oracle_feasible() -> None:
    """
    The function `test_lmi0_oracle_feasible` tests the feasibility of a Linear Matrix Inequality (LMI)
    oracle.
    """
    mat_f = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ]
    lmi0 = LMI0Oracle(mat_f)
    x = np.array([1.0, 0.0, 1.0])
    cut = lmi0.assess_feas(x)
    assert cut is None


def test_lmi0_oracle_infeasible() -> None:
    """
    The function `test_lmi0_oracle_infeasible` tests the case where the LMI oracle is infeasible.
    """
    mat_f = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ]
    lmi0 = LMI0Oracle(mat_f)
    x = np.array([-1.0, 0.0, -1.0])
    cut = lmi0.assess_feas(x)
    assert cut is not None
    assert np.allclose(cut[0], np.array([-1.0, -0.0, -0.0]))
    assert cut[1] == 1.0


def test_lmi0_oracle_infeasible2() -> None:
    """
    The function `test_lmi0_oracle_infeasible2` tests the case where the LMI oracle is infeasible.
    """
    mat_f = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ]
    lmi0 = LMI0Oracle(mat_f)
    x = np.array([1.0, 1.0, 1.0])
    cut = lmi0.assess_feas(x)
    assert cut is not None


def test_lmi0_oracle_infeasible3() -> None:
    """
    The function `test_lmi0_oracle_infeasible3` tests the case where the LMI oracle is infeasible.
    """
    mat_f = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ]
    lmi0 = LMI0Oracle(mat_f)
    x = np.array([1.0, -2.0, 1.0])
    cut = lmi0.assess_feas(x)
    assert cut is not None
    assert np.allclose(cut[0], np.array([-4.0, -4.0, -1.0]))
    assert cut[1] > 0.0
