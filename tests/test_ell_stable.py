import numpy as np
from ellalgo.ell_config import CutStatus
from pytest import approx

from ellalgo.ell_stable import EllStable


def test_construct():
    ell = EllStable(0.01, np.zeros(4))
    assert ell.no_defer_trick is False
    assert ell._kappa == 0.01
    assert ell._mq == approx(np.eye(4))
    assert ell._xc == approx(np.zeros(4))
    assert ell._ndim == 4


def test_update_central_cut():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), 0.0
    status = ell.update_central_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.01 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.1 * np.ones((4, 4)))
    assert ell._kappa == approx(0.16 / 15.0)


def test_update_bias_cut():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), 0.05
    status = ell.update_bias_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.03 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.2 * np.ones((4, 4)))
    assert ell._kappa == approx(0.008)


def test_update_parallel_central_cut():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [0.0, 0.05]
    status = ell.update_central_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.01 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.2 * np.ones((4, 4)))
    assert ell._kappa == approx(0.012)


def test_update_parallel():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [0.01, 0.04]
    status = ell.update_bias_cut(cut)
    assert status == CutStatus.Success
    assert ell._xc == approx(-0.0116 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.232 * np.ones((4, 4)))
    assert ell._kappa == approx(0.01232)


def test_update_parallel_no_effect():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [-0.04, 0.0625]
    status = ell.update_q(cut)
    assert status == CutStatus.NoEffect
    assert ell._xc == approx(np.zeros(4))
    assert ell._kappa == approx(0.01)
