from pytest import approx

from ellalgo.cutting_plane import CutStatus
from ellalgo.ell_calc import EllCalc


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
    ell_calc.calc_cc(0.1)
    assert ell_calc.sigma == approx(0.4)
    assert ell_calc.rho == approx(0.02)
    assert ell_calc.delta == approx(16.0 / 15.0)


def test_calc_dc():
    ell_calc = EllCalc(4)
    status = ell_calc.calc_dc(0.11, 0.1)
    assert status == CutStatus.NoSoln
    status = ell_calc.calc_dc(0.0, 0.1)
    assert status == CutStatus.Success
    status = ell_calc.calc_dc(-0.05, 0.1)
    assert status == CutStatus.NoEffect

    ell_calc.tsq = 0.01
    status = ell_calc.calc_dc(0.05, 0.1)
    assert status == CutStatus.Success
    assert ell_calc.sigma == approx(0.8)
    assert ell_calc.rho == approx(0.06)
    assert ell_calc.delta == approx(0.8)


def test_calc_ll_cc():
    ell_calc = EllCalc(4)
    ell_calc.tsq = 0.01
    status = ell_calc.calc_ll_cc(0.11)
    assert status == CutStatus.Success
    # Central cut
    assert ell_calc.sigma == approx(0.4)
    assert ell_calc.rho == approx(0.02)
    assert ell_calc.delta == approx(16.0 / 15.0)

    status = ell_calc.calc_ll_cc(0.05)
    assert status == CutStatus.Success
    assert ell_calc.sigma == approx(0.8)
    assert ell_calc.rho == approx(0.02)
    assert ell_calc.delta == approx(1.2)


def test_calc_ll():
    ell_calc = EllCalc(4)
    ell_calc.tsq = 0.01
    status = ell_calc.calc_ll(0.07, 0.03)
    assert status == CutStatus.NoSoln

    status = ell_calc.calc_ll(0.0, 0.05)
    assert status == CutStatus.Success
    assert ell_calc.sigma == approx(0.8)
    assert ell_calc.rho == approx(0.02)
    assert ell_calc.delta == approx(1.2)

    status = ell_calc.calc_ll(0.05, 0.11)
    assert status == CutStatus.Success
    assert ell_calc.sigma == approx(0.8)
    assert ell_calc.rho == approx(0.06)
    assert ell_calc.delta == approx(0.8)

    status = ell_calc.calc_ll(-0.07, 0.07)
    assert status == CutStatus.NoEffect

    status = ell_calc.calc_ll(0.01, 0.04)
    assert status == CutStatus.Success
    assert ell_calc.sigma == approx(0.8)
    assert ell_calc.rho == approx(0.02)
    assert ell_calc.delta == approx(1.232)

    status = ell_calc.calc_ll(-0.04, 0.0625)
    assert status == CutStatus.Success
    assert ell_calc.sigma == approx(3.950617283950619)
    assert ell_calc.rho == approx(0.04444444444444446)
    assert ell_calc.delta == approx(1.0)
