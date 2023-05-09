import numpy as np
from pytest import approx

from ellalgo.ell_stable import EllStable


def test_construct():
    ell = EllStable(0.01, np.zeros(4))
    assert ell.no_defer_trick is False
    assert ell._kappa == 0.01
    assert ell._mq == approx(np.eye(4))
    assert ell._xc == approx(np.zeros(4))
    assert ell._n == 4


def test_update_cc():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), 0.0
    ell.update_cc(cut)
    # omega = 1.0
    # assert ell.sigma == approx(0.4)
    # assert ell.rho == approx(0.02)
    # assert ell.delta == approx(16.0 / 15.0)
    assert ell._xc == approx(-0.01 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.1 * np.ones((4, 4)))
    assert ell._kappa == approx(0.16 / 15.0)


def test_update_dc():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), 0.05
    ell.update_dc(cut)

    # assert ell.sigma == approx(0.8)
    # assert ell.rho == approx(0.06)
    # assert ell.delta == approx(0.8)
    assert ell._xc == approx(-0.03 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.2 * np.ones((4, 4)))
    assert ell._kappa == approx(0.008)


def test_update_ll_cc():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [0.0, 0.05]
    ell.update_cc(cut)
    # assert ell.sigma == approx(0.8)
    # assert ell.rho == approx(0.02)
    # assert ell.delta == approx(1.2)
    assert ell._xc == approx(-0.01 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.2 * np.ones((4, 4)))
    assert ell._kappa == approx(0.012)


def test_calc_ll():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [0.01, 0.04]
    ell.update_dc(cut)

    # assert ell.sigma == approx(0.928)
    # assert ell.rho == approx(0.0232)
    # assert ell.delta == approx(1.232)
    assert ell._xc == approx(-0.0116 * np.ones(4))
    # assert ell._mq == approx(np.eye(4) - 0.232 * np.ones((4, 4)))
    assert ell._kappa == approx(0.01232)


def test_calc_ll_no_effect():
    ell = EllStable(0.01, np.zeros(4))
    cut = 0.5 * np.ones(4), [-0.04, 0.0625]
    ell.update_dc(cut)
    # assert ell.sigma == approx(0.0)
    # assert ell.rho == approx(0.0)
    # assert ell.delta == approx(1.0)
    assert ell._xc == approx(np.zeros(4))
    # assert ell._mq == approx(np.eye(4))
    assert ell._kappa == approx(0.01)
