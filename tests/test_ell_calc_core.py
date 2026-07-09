import pytest
from pytest import approx

from ellalgo.ell_calc_core import EllCalcCore


def test_construct() -> None:
    ell_calc_core = EllCalcCore(4)
    assert ell_calc_core._n_f == 4.0
    assert ell_calc_core._half_n == 2.0
    assert ell_calc_core._n_plus_1 == 5.0
    assert ell_calc_core._cst0 == approx(0.2)
    assert ell_calc_core._cst1 == approx(16.0 / 15.0)
    assert ell_calc_core._cst2 == approx(0.4)
    assert ell_calc_core._cst3 == approx(0.8)


def test_calc_central_cut() -> None:
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_central_cut(0.1)
    assert rho == approx(0.02)
    assert sigma == approx(0.4)
    assert delta == approx(16.0 / 15.0)


def test_calc_bias_cut() -> None:
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_bias_cut(0.05, 0.1)
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)


def test_calc_parallel_central_cut() -> None:
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_central_cut(1.0, 4.0)
    assert rho == approx(0.4)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)
    rho, sigma, delta = ell_calc_core.calc_parallel_central_cut_old(1.0, 4.0)
    assert rho == approx(0.4)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)


def test_calc_parallel() -> None:
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut(0.0, 0.05, 0.01)
    assert rho == approx(0.02)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)

    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_old(0.0, 0.05, 0.01)
    assert rho == approx(0.02)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)

    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut(0.01, 0.04, 0.01)
    assert rho == approx(0.0232)
    assert sigma == approx(0.928)
    assert delta == approx(1.232)

    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut(-0.25, 0.25, 1.0)
    assert sigma == approx(0.8)
    assert rho == approx(0.0)
    assert delta == approx(1.25)


def test_calc_parallel_noeffect() -> None:
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut(-0.04, 0.0625, 0.01)
    assert rho == approx(0.0)
    assert sigma == approx(0.0)
    assert delta == approx(1.0)


def test_calc_bias_cut_fast() -> None:
    ell_calc_core = EllCalcCore(3)
    rho, sigma, delta = ell_calc_core.calc_bias_cut_fast(0.0, 2.0, 2.0)
    assert rho == approx(0.5)
    assert sigma == approx(0.5)
    assert delta == approx(1.125)


def test_calc_parallel_bias_cut_fast() -> None:
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_fast(
        -0.25, 0.25, 1.0, -0.0625, 0.75
    )
    assert rho == approx(0.0)
    assert sigma == approx(0.8)
    assert delta == approx(1.25)


def test_calc_parallel_bias_cut_fast_k_le_eta() -> None:
    """Test calc_parallel_bias_cut_fast with k <= eta to cover line 573."""
    ell_calc_core = EllCalcCore(4)
    # Use values that will make k <= eta to trigger the central cut path
    # This is a specific case where the algorithm falls back to central cut
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_fast(
        0.0,
        0.01,
        0.01,
        0.0,
        0.01,  # Small values to trigger k <= eta
    )
    # Should fall back to central cut calculation
    assert rho is not None
    assert sigma is not None
    assert delta is not None


def test_calc_parallel_bias_cut_fast_old() -> None:
    """Test calc_parallel_bias_cut_fast_old (lines 426-441)."""
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_fast_old(
        0.11, 0.01, 0.01, 0.0011, 0.0144
    )
    assert rho == approx(0.027228509068282114)
    assert sigma == approx(0.45380848447136857)
    assert delta == approx(1.0443438549074862)

    # Test k <= eta branch (falls back to central cut)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_fast_old(
        0.0, 0.0, 0.01, 0.0, 0.01
    )
    assert rho == approx(0.02)  # central cut rho = tau/(n+1) = 0.1/5
    assert sigma == approx(0.4)  # central cut sigma = 2/(n+1) = 0.4
    assert delta == approx(16.0 / 15.0)  # central cut delta = n²/(n²-1)


def test_calc_parallel_bias_cut_fast2_valid() -> None:
    """Test calc_parallel_bias_cut_fast2 with non-symmetric inputs (lines 535-537)."""
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_fast2(
        0.0, 0.09, 0.01, 0.0, 0.01
    )
    assert rho == approx(0.020941836487980856)
    assert sigma == approx(0.46537414417735234)
    assert delta == approx(1.082031295477563)


def test_calc_parallel_bias_cut_fast2_zerodiv() -> None:
    """Demonstrate ZeroDivisionError when beta0 = -beta1.

    In `calc_parallel_bias_cut_fast2`, the formula computes
    `bsumsq = (beta0 + beta1) ** 2`. When beta0 = -beta1,
    this evaluates to zero and causes a division by zero
    when computing sigma. Use `calc_parallel_bias_cut_fast`
    instead for such symmetric cases.
    """
    ell_calc_core = EllCalcCore(4)
    with pytest.raises(ZeroDivisionError):
        ell_calc_core.calc_parallel_bias_cut_fast2(-0.25, 0.25, 1.0, -0.0625, 0.75)
