from pytest import approx

from ellalgo.ell_calc_core import EllCalcCore


def test_construct():
    ell_calc_core = EllCalcCore(4)
    assert ell_calc_core._n_f == 4.0
    assert ell_calc_core._half_n == 2.0
    assert ell_calc_core._n_plus_1 == 5.0
    assert ell_calc_core._cst0 == approx(0.2)
    assert ell_calc_core._cst1 == approx(16.0 / 15.0)
    assert ell_calc_core._cst2 == approx(0.4)
    assert ell_calc_core._cst3 == approx(0.8)


def test_calc_central_cut():
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_central_cut(0.1)
    assert rho == approx(0.02)
    assert sigma == approx(0.4)
    assert delta == approx(16.0 / 15.0)


def test_calc_bias_cut():
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_bias_cut(0.05, 0.1)
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)


def test_calc_parallel_central_cut():
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_central_cut(1.0, 4.0)
    assert rho == approx(0.4)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)
    rho, sigma, delta = ell_calc_core.calc_parallel_central_cut_old(1.0, 4.0)
    assert rho == approx(0.4)
    assert sigma == approx(0.8)
    assert delta == approx(1.2)


def test_calc_parallel():
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


def test_calc_parallel_noeffect():
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut(-0.04, 0.0625, 0.01)
    assert rho == approx(0.0)
    assert sigma == approx(0.0)
    assert delta == approx(1.0)


def test_calc_bias_cut_q():
    ell_calc_q = EllCalcCore(4)
    rho, sigma, delta = ell_calc_q.calc_bias_cut(0.05, 0.1)
    assert rho == approx(0.06)
    assert sigma == approx(0.8)
    assert delta == approx(0.8)


def test_calc_parallel_bias_cut_q():
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut(0.0, 0.05, 0.01)
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


def test_calc_bias_cut_fast():
    ell_calc_core = EllCalcCore(3)
    rho, sigma, delta = ell_calc_core.calc_bias_cut_fast(0.0, 2.0, 2.0)
    assert rho == approx(0.5)
    assert sigma == approx(0.5)
    assert delta == approx(1.125)


def test_calc_parallel_bias_cut_fast():
    ell_calc_core = EllCalcCore(4)
    rho, sigma, delta = ell_calc_core.calc_parallel_bias_cut_fast(
        -0.25, 0.25, 1.0, -0.0625, 0.75
    )
    assert rho == approx(0.0)
    assert sigma == approx(0.8)
    assert delta == approx(1.25)
