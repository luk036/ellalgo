import numpy as np
from pytest import approx

from ellalgo.ell_config import CutStatus
from ellalgo.ell_stable import EllStable


def test_construct() -> None:
    ell = EllStable(0.01, np.zeros(4))
    assert ell.no_defer_trick is False
    assert ell._kappa == 0.01
    assert ell._mq == approx(np.eye(4))
    assert ell._xc == approx(np.zeros(4))
    assert ell._tsq == 0.0


def test_update_central_cut() -> None:
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), 0.0
    status = ell.update_central_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.01 * np.ones(4))
    assert ell._kappa == approx(0.16 / 15.0)
    assert ell._tsq == 0.01


def test_update_bias_cut() -> None:
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), 0.05
    status = ell.update_bias_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.03 * np.ones(4))
    assert ell._kappa == approx(0.008)
    assert ell._tsq == 0.01


def test_update_parallel_central_cut() -> None:
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [0.0, 0.05]
    status = ell.update_central_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.01 * np.ones(4))
    assert ell._kappa == approx(0.012)
    assert ell._tsq == 0.01


def test_update_parallel() -> None:
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [0.01, 0.04]
    status = ell.update_bias_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.0116 * np.ones(4))
    assert ell._kappa == approx(0.01232)
    assert ell._tsq == 0.01


def test_update_parallel_no_effect() -> None:
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [-0.04, 0.0625]
    status = ell.update_bias_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(np.zeros(4))
    assert ell._mq == approx(np.eye(4))
    assert ell._kappa == approx(0.01)


def test_update_q_no_effect() -> None:
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [-0.04, 0.0625]
    status = ell.update_q(cut)
    assert status == CutStatus.NoEffect
    assert ell._xc == approx(np.zeros(4))
    assert ell._mq == approx(np.eye(4))
    assert ell._kappa == approx(0.01)


def test_update_central_cut_with_no_defer_trick() -> None:
    """Test update_central_cut with no_defer_trick=True to cover lines 234-235."""
    ell = EllStable(0.01, np.zeros(4))
    ell.no_defer_trick = True  # Enable no_defer_trick
    cut = 0.5 * np.ones(4), 0.0
    status = ell.update_central_cut(cut)
    assert status == CutStatus.Success
    # When no_defer_trick is True, _mq should be multiplied by _kappa and _kappa set to 1.0
    assert ell._kappa == approx(1.0)  # Should be reset to 1.0
