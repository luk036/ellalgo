import numpy as np

from ellalgo.ell import Ell


def test_ell_initialization_with_list() -> None:
    """Test ellipsoid initialization with a list of values."""
    val = np.array([1.0, 2.0, 3.0])
    xc = np.array([0.0, 0.0, 0.0])
    ell = Ell(val, xc)
    assert ell._kappa == 1.0
    assert np.array_equal(ell._mq, np.diag(val))
    assert np.array_equal(ell.xc(), xc)
    assert ell.tsq() == 0.0


def test_ell_xc_setter_and_getter() -> None:
    """Test the xc getter and setter methods."""
    ell = Ell(1.0, np.array([0.0, 0.0]))
    new_xc = np.array([1.0, 2.0])
    ell.set_xc(new_xc)
    assert np.array_equal(ell.xc(), new_xc)


def test_ell_tsq_getter() -> None:
    """Test the tsq getter method."""
    ell = Ell(1.0, np.array([0.0, 0.0]))
    assert ell.tsq() == 0.0


def test_ell_no_defer_trick() -> None:
    """Test the no_defer_trick functionality."""
    ell = Ell(1.0, np.array([0.0, 0.0]))
    ell.no_defer_trick = True
    cut = (np.array([1.0, 0.0]), 0.0)
    ell.update_central_cut(cut)
    assert ell._kappa == 1.0
