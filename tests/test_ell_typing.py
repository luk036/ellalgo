"""Tests for ell_typing.py default method implementations."""

import numpy as np

from ellalgo.ell import Ell
from ellalgo.ell_config import CutStatus
from ellalgo.ell_typing import OracleFeas, SearchSpace


class _ConcreteOracleFeas(OracleFeas):
    """Minimal concrete implementation for testing OracleFeas default methods."""

    def __init__(self, mat_f=None, mat_b=None):  # noqa: D107
        pass

    def assess_feas(self, x_center):  # noqa: D102
        return None


class _ConcreteSearchSpace(SearchSpace):
    """Minimal concrete implementation for testing SearchSpace default methods."""

    def __init__(self, val, x_center):  # noqa: D107
        pass

    def update_bias_cut(self, cut):  # noqa: D102
        return CutStatus.Success

    def update_central_cut(self, cut):  # noqa: D102
        return CutStatus.Success

    def xc(self):  # noqa: D102
        return np.zeros(2)

    def tsq(self):  # noqa: D102
        return 0.0


def test_oracle_feas_update_default() -> None:
    """Test OracleFeas.update() default no-op (line 53)."""
    oracle = _ConcreteOracleFeas()
    oracle.update(1.0)


def test_search_space_update_q_default() -> None:
    """Test SearchSpace.update_q() default delegates to update_bias_cut (line 150)."""
    space = _ConcreteSearchSpace(None, None)
    result = space.update_q((np.ones(2), 0.0))
    assert result == CutStatus.Success


def test_search_space_set_xc_default() -> None:
    """Test SearchSpace.set_xc() default no-op (line 162)."""
    space = _ConcreteSearchSpace(None, None)
    space.set_xc(np.array([1.0, 2.0]))


def test_ell_update_q_delegates() -> None:
    """Test Ell.update_q() inherits default from SearchSpace, delegates to update_bias_cut."""
    ell = Ell(1.0, np.zeros(2))
    cut = (np.array([0.5, 0.5]), 0.0)
    status = ell.update_q(cut)
    assert status == CutStatus.Success
