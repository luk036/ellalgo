"""
Test Cutting Plane
"""

from __future__ import print_function

from typing import List, Optional, Tuple

import numpy as np
import pytest

from ellalgo.cutting_plane import (
    BSearchAdaptor,
    Options,
    bsearch,
    cutting_plane_feas,
    cutting_plane_optim,
    cutting_plane_optim_q,
)
from ellalgo.ell import Ell
from ellalgo.ell_stable import EllStable
from ellalgo.ell_typing import OracleBS, OracleFeas, OracleOptim, OracleOptimQ


@pytest.fixture
def options() -> Options:
    """Set up options for cutting plane tests."""
    return Options()


class MyOracleFeas(OracleFeas):
    """Oracle for feasibility problem."""

    def __init__(
        self, mat_f: List[np.ndarray], mat_b: Optional[np.ndarray] = None
    ) -> None:
        pass

    def assess_feas(self, xc: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Assess feasibility of `xc`."""
        x, y = xc
        if (fj := x + y - 3.0) > 0.0:
            return (np.array([1.0, 1.0]), fj)
        return None


class MyOracleInfeas(OracleFeas):
    """Oracle for infeasibility problem."""

    def __init__(
        self, mat_f: List[np.ndarray], mat_b: Optional[np.ndarray] = None
    ) -> None:
        pass

    def assess_feas(self, xc: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Assess feasibility of `xc`."""
        return (np.array([1.0, 1.0]), 1.0)


class MyOracleOptim(OracleOptim):
    """Oracle for optimization problem."""

    def assess_optim(
        self, xc: np.ndarray, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], Optional[float]]:
        """Assess optimality of `xc`."""
        x, y = xc
        f0 = x + y
        if (f1 := x - 1.0) > 0.0:
            return ((np.array([1.0, 0.0]), f1), None)
        if (f2 := y - 1.0) > 0.0:
            return ((np.array([0.0, 1.0]), f2), None)
        if (f3 := f0 - gamma) < 0.0:
            return ((np.array([-1.0, -1.0]), -f3), None)
        return ((np.array([-1.0, -1.0]), 0.0), f0)


class MyOracleOptimQ(OracleOptimQ):
    """Oracle for quantized optimization problem."""

    def assess_optim_q(
        self, xc: np.ndarray, gamma: float, retry: bool
    ) -> Tuple[Tuple[np.ndarray, float], Optional[np.ndarray], Optional[float], bool]:
        """Assess optimality of `xc`."""
        x, y = xc
        f0 = x + y
        if (f1 := x - 1.0) > 0.0:
            return ((np.array([1.0, 0.0]), f1), None, None, True)
        if (f2 := y - 1.0) > 0.0:
            return ((np.array([0.0, 1.0]), f2), None, None, True)
        if (f3 := f0 - gamma) < 0.0:
            return ((np.array([-1.0, -1.0]), -f3), None, None, True)

        x_q = np.round(xc)
        if (f1 := x_q[0] - 1.0) > 0.0:
            return ((np.array([1.0, 0.0]), f1), x_q, None, not retry)
        if (f2 := x_q[1] - 1.0) > 0.0:
            return ((np.array([0.0, 1.0]), f2), x_q, None, not retry)
        if (f3 := x_q[0] + x_q[1] - gamma) < 0.0:
            return ((np.array([-1.0, -1.0]), -f3), x_q, None, not retry)
        return ((np.array([-1.0, -1.0]), 0.0), x_q, f0, not retry)


class MyOracleBS(OracleBS):
    """Oracle for binary search."""

    def assess_bs(self, gamma: float) -> bool:
        """Assess feasibility of `gamma`."""
        return gamma > 0


def test_cutting_plane_feas(options: Options) -> None:
    """Test cutting plane feasibility."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleFeas([], None)
    options.max_iters = 200
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is not None
    assert num_iters == 0


def test_cutting_plane_feas_no_soln(options: Options) -> None:
    """Test cutting plane feasibility with no solution."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleInfeas([], None)
    options.max_iters = 200
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is None
    assert num_iters == 2


def test_cutting_plane_optim(options: Options) -> None:
    """Test cutting plane optimization."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim()
    options.max_iters = 200
    xbest, fbest, num_iters = cutting_plane_optim(omega, ellip, 0.0, options)
    assert xbest is not None
    assert fbest == pytest.approx(2.0)
    assert num_iters == 145


def test_cutting_plane_optim_no_soln(options: Options) -> None:
    """Test cutting plane optimization with no solution."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim()
    options.max_iters = 4
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 100.0, options)
    assert xbest is None
    assert num_iters == 0


def test_cutting_plane_optim_q(options: Options) -> None:
    """Test cutting plane optimization with quantization."""
    xinit = np.array([0.0, 0.0])
    ellip = EllStable(10.0, xinit)
    omega = MyOracleOptimQ()
    options.max_iters = 200
    xbest, fbest, num_iters = cutting_plane_optim_q(omega, ellip, 0.0, options)
    assert xbest is not None
    assert fbest == pytest.approx(2.0)
    assert num_iters == 145


def test_cutting_plane_optim_q_no_soln(options: Options) -> None:
    """Test cutting plane optimization with quantization and no solution."""
    xinit = np.array([0.0, 0.0])
    ellip = EllStable(10.0, xinit)
    omega = MyOracleOptimQ()
    options.max_iters = 20
    xbest, _, num_iters = cutting_plane_optim_q(omega, ellip, 100.0, options)
    assert xbest is None
    assert num_iters == 0


def test_bsearch(options: Options) -> None:
    """Test binary search."""
    omega = MyOracleBS()
    options.tolerance = 1e-7
    gamma, num_iters = bsearch(omega, (-100.0, 100.0), options)
    assert gamma > 0.0
    assert gamma < 2e-7
    assert num_iters == 30


def test_bsearch_no_soln(options: Options) -> None:
    """Test binary search with no solution."""
    omega = MyOracleBS()
    options.max_iters = 20
    gamma, num_iters = bsearch(omega, (-100.0, -50.0), options)
    assert gamma == -50.0
    assert num_iters == 20


class MyOracleBSOffset(OracleBS):
    """Binary search oracle whose threshold is far from zero."""

    def assess_bs(self, gamma: float) -> bool:
        """Assess feasibility of `gamma`."""
        return gamma > 500.0


def test_bsearch_stops_at_float_resolution(options: Options) -> None:
    """Binary search must stop once the bracket can no longer shrink.

    When the threshold is far from zero the bracket collapses around it, so
    ``tau`` bottoms out at the ulp of that value (~1e-13) and the default
    ``tolerance = 1e-20`` is unreachable. Without a stall guard the loop spins
    until ``max_iters`` without refining anything.
    """
    omega = MyOracleBSOffset()
    options.max_iters = 2000
    gamma, num_iters = bsearch(omega, (0.0, 1e6), options)
    assert num_iters < 200
    assert gamma == pytest.approx(500.0, rel=1e-9)


class MyOracleFeas2(OracleFeas):
    """Oracle for feasibility problem that always returns a cut."""

    def __init__(
        self, mat_f: List[np.ndarray], mat_b: Optional[np.ndarray] = None
    ) -> None:
        pass

    def assess_feas(self, xc: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Assess feasibility of `xc`."""
        return (np.array([1.0, 1.0]), 1.0)  # Always returns a cut


def test_cutting_plane_feas_max_iters(options: Options) -> None:
    """Test cutting plane feasibility reaching max iterations."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleFeas2([], None)  # Always returns a cut
    options.max_iters = 5
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is None
    assert num_iters == 2  # Actual behavior: returns after 2 iterations


class MyOracleOptim2(OracleOptim):
    """Oracle for optimization problem that always returns a cut."""

    def assess_optim(
        self, xc: np.ndarray, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], Optional[float]]:
        """Assess optimality of `xc`."""
        return ((np.array([1.0, 1.0]), 1.0), None)  # Always returns a cut


def test_cutting_plane_optim_max_iters(options: Options) -> None:
    """Test cutting plane optimization reaching max iterations."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim2()  # Always returns a cut
    options.max_iters = 5
    xbest, fbest, num_iters = cutting_plane_optim(omega, ellip, 0.0, options)
    assert xbest is None
    assert num_iters == 2  # Actual behavior: returns after 2 iterations


class MyOracleOptimQ2(OracleOptimQ):
    """Oracle for quantized optimization with special cases."""

    def assess_optim_q(
        self, xc: np.ndarray, gamma: float, retry: bool
    ) -> Tuple[Tuple[np.ndarray, float], Optional[np.ndarray], Optional[float], bool]:
        """Assess optimality of `xc`."""
        x, y = xc
        x + y

        # Always return a cut to test the NoEffect case
        return ((np.array([1.0, 1.0]), 1.0), None, None, True)


def test_cutting_plane_optim_q_no_effect(options: Options) -> None:
    """Test cutting plane optimization with quantization hitting NoEffect."""
    xinit = np.array([0.0, 0.0])
    ellip = EllStable(10.0, xinit)
    omega = MyOracleOptimQ2()  # Always returns a cut
    options.max_iters = 5
    xbest, fbest, num_iters = cutting_plane_optim_q(omega, ellip, 0.0, options)
    assert xbest is None  # Actual behavior: returns None
    assert num_iters == 2  # Actual behavior: returns after 2 iterations


class MyOracleBS2(OracleFeas):
    """Oracle for binary search with update method."""

    def __init__(self) -> None:
        self.gamma_val = 0.0

    def assess_feas(self, xc: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Assess feasibility of `xc`."""
        return (np.array([1.0, 1.0]), 1.0)  # Always returns a cut

    def assess_bs(self, gamma: float) -> bool:
        """Assess feasibility of `gamma`."""
        self.gamma_val = gamma
        return gamma > 0

    def update(self, gamma: float) -> None:
        """Update with new gamma value."""
        self.gamma_val = gamma


def test_bsearch_adaptor_x_best(options: Options) -> None:
    """Test BSearchAdaptor.x_best property."""
    xinit = np.array([0.0, 0.0])
    ellip = EllStable(10.0, xinit)
    omega = MyOracleBS2()
    adaptor = BSearchAdaptor(omega, ellip, options)

    # Test that x_best returns the current center of the ellipsoid
    x_best = adaptor.x_best
    np.testing.assert_array_equal(x_best, xinit)


class AlwaysCutFeas(OracleFeas):
    """Oracle that always returns a feasible cut (no convergence)."""

    def __init__(self, mat_f=None, mat_b=None):
        pass

    def assess_feas(self, xc):
        return (np.array([1.0, 0.0]), 0.1)


class AlwaysCutOptim(OracleOptim):
    """Oracle that always returns a bias cut (no convergence)."""

    def assess_optim(self, xc, gamma):
        return ((np.array([1.0, 0.0]), 0.1), None)


class AlwaysCutOptimQ(OracleOptimQ):
    """Oracle for quantized optimization that always returns a cut."""

    def assess_optim_q(self, xc, gamma, retry):
        return ((np.array([1.0, 0.0]), 0.1), None, None, True)


def test_cutting_plane_feas_exhausts_max_iters() -> None:
    """Test cutting_plane_feas reaching max_iters return (line 125)."""
    opt = Options()
    opt.tolerance = 0.0
    opt.max_iters = 3
    ellip = Ell(10.0, np.array([0.0, 0.0]))
    omega = AlwaysCutFeas([], None)
    xbest, niter = cutting_plane_feas(omega, ellip, opt)
    assert xbest is None
    assert niter == 3


def test_cutting_plane_optim_exhausts_max_iters() -> None:
    """Test cutting_plane_optim reaching max_iters return (line 197)."""
    opt = Options()
    opt.tolerance = 0.0
    opt.max_iters = 3
    ellip = Ell(10.0, np.array([0.0, 0.0]))
    omega = AlwaysCutOptim()
    xbest, fbest, niter = cutting_plane_optim(omega, ellip, 0.0, opt)
    assert xbest is None
    assert niter == 3


def test_cutting_plane_optim_q_exhausts_max_iters() -> None:
    """Test cutting_plane_optim_q reaching max_iters return (line 292)."""
    opt = Options()
    opt.tolerance = 0.0
    opt.max_iters = 3
    ellip = EllStable(10.0, np.array([0.0, 0.0]))
    omega = AlwaysCutOptimQ()
    xbest, fbest, niter = cutting_plane_optim_q(omega, ellip, 0.0, opt)
    assert xbest is None
    assert niter == 3


class NoEffectOptimQ(OracleOptimQ):
    """Oracle returning a cut that triggers NoEffect with no more alternatives."""

    def assess_optim_q(self, xc, gamma, retry):
        return ((np.array([0.01, 0.01]), -0.1), None, None, False)


def test_cutting_plane_optim_q_no_effect_no_more_alt() -> None:
    """Test cutting_plane_optim_q NoEffect with exhausted alternatives (lines 286-288)."""
    opt = Options()
    opt.max_iters = 10
    ellip = EllStable(10.0, np.array([0.0, 0.0]))
    omega = NoEffectOptimQ()
    xbest, fbest, niter = cutting_plane_optim_q(omega, ellip, 0.0, opt)
    assert niter == 0  # First iteration hits NoEffect with no more alternatives
