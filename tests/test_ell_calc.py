from pytest import approx

from ellalgo.ell_calc import EllCalc
from ellalgo.ell_config import CutStatus


def test_construct() -> None:
    ell_calc = EllCalc(4)
    assert ell_calc.use_parallel_cut is True
    assert ell_calc._n_f == 4.0


def test_calc_central_cut() -> None:
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_single_or_parallel_central_cut([0, 0.05], 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)


def test_calc_bias_cut() -> None:
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_bias_cut(0.11, 0.01)
    assert status == CutStatus.NoSoln
    assert result is None
    status, result = ell_calc.calc_bias_cut(0.01, 0.01)
    assert status == CutStatus.Success
    assert result is not None

    status, result = ell_calc.calc_bias_cut(0.05, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)


def test_calc_parallel_central_cut() -> None:
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_single_or_parallel_central_cut([0, 0.05], 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.02)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)


def test_calc_parallel() -> None:
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_parallel(0.07, 0.03, 0.01)
    assert status == CutStatus.NoSoln
    assert result is None

    status, result = ell_calc.calc_parallel(0.0, 0.05, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.02)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)

    status, result = ell_calc.calc_parallel(0.05, 0.11, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)

    status, result = ell_calc.calc_parallel(0.01, 0.04, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.0232)
    assert sigma == approx(0.928)
    assert delta == approx(1.232)


def test_calc_bias_cut_q() -> None:
    ell_calc_q = EllCalc(4)
    status, result = ell_calc_q.calc_bias_cut_q(0.11, 0.01)
    assert status == CutStatus.NoSoln
    assert result is None
    status, result = ell_calc_q.calc_bias_cut_q(0.01, 0.01)
    assert status == CutStatus.Success
    status, result = ell_calc_q.calc_bias_cut_q(-0.05, 0.01)
    assert status == CutStatus.NoEffect
    assert result is None

    status, result = ell_calc_q.calc_bias_cut_q(0.05, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)


def test_calc_parallel_q() -> None:
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_parallel_q(0.07, 0.03, 0.01)
    assert status == CutStatus.NoSoln
    assert result is None
    status, result = ell_calc.calc_parallel_q(-0.04, 0.0625, 0.01)
    assert status == CutStatus.NoEffect
    assert result is None

    status, result = ell_calc.calc_parallel_q(0.0, 0.05, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.02)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)

    status, result = ell_calc.calc_parallel_q(0.05, 0.11, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)

    status, result = ell_calc.calc_parallel_q(0.01, 0.04, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.0232)
    assert sigma == approx(0.928)
    assert delta == approx(1.232)


def test_calc_single_or_parallel_central_cut_tsq_le_b1sq() -> None:
    """Test case where tsq <= b1sq to cover line 157."""
    ell_calc = EllCalc(4)
    # Use small tsq and larger beta[1] to trigger tsq <= b1sq condition
    status, result = ell_calc.calc_single_or_parallel_central_cut(
        [0.0, 0.1], 0.005
    )  # tsq=0.005, b1sq=0.01
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    # This should go through the central cut path


def test_calc_single_or_parallel_q_no_parallel_cut() -> None:
    """Test case where use_parallel_cut is False to cover line 255."""
    ell_calc = EllCalc(4)
    ell_calc.use_parallel_cut = False  # Disable parallel cuts
    # This should trigger the len(beta) < 2 or not self.use_parallel_cut condition
    status, result = ell_calc.calc_single_or_parallel_q([0.05, 0.1], 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    # This should go through the bias_cut_q path
