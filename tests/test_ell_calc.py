from pytest import approx

from ellalgo.ell_calc import EllCalc
from ellalgo.ell_config import CutStatus


def test_construct():
    ell_calc = EllCalc(4)
    assert ell_calc.use_parallel_cut is True
    assert ell_calc._n_f == 4.0


def test_calc_central_cut():
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_single_or_parallel_central_cut([0, 0.05], 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)


def test_calc_bias_cut():
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


def test_calc_parallel_central_cut():
    ell_calc = EllCalc(4)
    status, result = ell_calc.calc_single_or_parallel_central_cut([0, 0.05], 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert rho == approx(0.02)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)


def test_calc_parallel():
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


def test_calc_bias_cut_q():
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


def test_calc_parallel_q():
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
