from pytest import approx

from ellalgo.ell_config import CutStatus
from ellalgo.ell_calc import EllCalc, EllCalcQ


def test_construct():
    ell_calc = EllCalc(4)
    assert ell_calc.use_parallel_cut is True
    assert ell_calc.n_f == 4.0
    assert ell_calc.half_n == 2.0
    assert ell_calc.cst0 == approx(0.2)
    assert ell_calc.cst1 == approx(16.0 / 15.0)
    assert ell_calc.cst2 == approx(0.4)
    assert ell_calc.cst3 == approx(0.8)


def test_calc_cc():
    ell_calc = EllCalc(4)
    status, rho, sigma, delta = ell_calc.calc_cc(0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.4)
    assert rho == approx(0.02)
    assert delta == approx(16.0 / 15.0)


def test_calc_dc():
    ell_calc = EllCalc(4)
    status, rho, sigma, delta = ell_calc.calc_dc(0.11, 0.01)
    assert status == CutStatus.NoSoln
    status, rho, sigma, delta = ell_calc.calc_dc(0.01, 0.01)
    assert status == CutStatus.Success
    # status, rho, sigma, delta = ell_calc.calc_dc(-0.05)
    # assert status == CutStatus.NoEffect

    status, rho, sigma, delta = ell_calc.calc_dc(0.05, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.06)
    assert delta == approx(0.8)


def test_calc_ll_cc():
    ell_calc = EllCalc(4)
    status, rho, sigma, delta = ell_calc.calc_ll_cc(0.11, 0.01)
    assert status == CutStatus.Success
    # Central cut
    assert sigma == approx(0.4)
    assert rho == approx(0.02)
    assert delta == approx(16.0 / 15.0)

    status, rho, sigma, delta = ell_calc.calc_ll_cc(0.05, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)


def test_calc_ll():
    ell_calc = EllCalc(4)
    status, rho, sigma, delta = ell_calc.calc_ll(0.07, 0.03, 0.01)
    assert status == CutStatus.NoSoln

    status, rho, sigma, delta = ell_calc.calc_ll(0.0, 0.05, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)

    status, rho, sigma, delta = ell_calc.calc_ll(0.05, 0.11, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.06)
    assert delta == approx(0.8)

    # status, rho, sigma, delta = ell_calc.calc_ll(-0.07, 0.07)
    # assert status == CutStatus.NoEffect

    status, rho, sigma, delta = ell_calc.calc_ll(0.01, 0.04, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.928)
    assert rho == approx(0.0232)
    assert delta == approx(1.232)


def test_calc_ll_noeffect():
    ell_calc = EllCalc(4)

    status, rho, sigma, delta = ell_calc.calc_ll(-0.04, 0.0625, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.0)
    assert rho == approx(0.0)
    assert delta == approx(1.0)


def test_calc_dc_q():
    ell_calc_q = EllCalcQ(4)
    status, rho, sigma, delta = ell_calc_q.calc_dc_q(0.11, 0.01)
    assert status == CutStatus.NoSoln
    status, rho, sigma, delta = ell_calc_q.calc_dc_q(0.01, 0.01)
    assert status == CutStatus.Success
    status, rho, sigma, delta = ell_calc_q.calc_dc_q(-0.05, 0.01)
    assert status == CutStatus.NoEffect

    status, rho, sigma, delta = ell_calc_q.calc_dc_q(0.05, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.06)
    assert delta == approx(0.8)


def test_calc_ll_q():
    ell_calc_q = EllCalcQ(4)
    status, rho, sigma, delta = ell_calc_q.calc_ll_q(0.07, 0.03, 0.01)
    assert status == CutStatus.NoSoln

    status, rho, sigma, delta = ell_calc_q.calc_ll_q(0.0, 0.05, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)

    status, rho, sigma, delta = ell_calc_q.calc_ll_q(0.05, 0.11, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.8)
    assert rho == approx(0.06)
    assert delta == approx(0.8)

    # status, rho, sigma, delta = ell_calc.calc_ll(-0.07, 0.07)
    # assert status == CutStatus.NoEffect

    status, rho, sigma, delta = ell_calc_q.calc_ll_q(0.01, 0.04, 0.01)
    assert status == CutStatus.Success
    assert sigma == approx(0.928)
    assert rho == approx(0.0232)
    assert delta == approx(1.232)

    status, rho, sigma, delta = ell_calc_q.calc_ll_q(-0.0100000001, 0.0100000002, 0.01)
    assert sigma == approx(-24.8)
    assert rho == approx(-1.24e-9)
    assert delta == approx(1.32)
