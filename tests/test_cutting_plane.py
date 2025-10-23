"""
Test Cutting Plane
"""

from __future__ import print_function

from typing import Optional, Tuple

import numpy as np
import pytest

from ellalgo.cutting_plane import (
    Options,
    bsearch,
    cutting_plane_feas,
    cutting_plane_optim,
    cutting_plane_optim_q,
)
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleBS, OracleFeas, OracleOptim, OracleOptimQ


@pytest.fixture
def options():
    """Set up options for cutting plane tests."""
    return Options()


class MyOracleFeas(OracleFeas):
    """Oracle for feasibility problem."""

    def assess_feas(self, xc) -> Optional[Tuple[np.ndarray, float]]:
        """Assess feasibility of `xc`."""
        x, y = xc
        if (fj := x + y - 3.0) > 0.0:
            return (np.array([1.0, 1.0]), fj)
        return None


class MyOracleInfeas(OracleFeas):
    """Oracle for infeasibility problem."""

    def assess_feas(self, xc) -> Optional[Tuple[np.ndarray, float]]:
        """Assess feasibility of `xc`."""
        return (np.array([1.0, 1.0]), 1.0)


class MyOracleOptim(OracleOptim):
    """Oracle for optimization problem."""

    def assess_optim(
        self, xc, gamma: float
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

    def assess_optim_q(self, xc, gamma: float, retry: bool):
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


def test_cutting_plane_feas(options):
    """Test cutting plane feasibility."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleFeas()
    options.max_iters = 200
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is not None
    assert num_iters == 0


def test_cutting_plane_feas_no_soln(options):
    """Test cutting plane feasibility with no solution."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleInfeas()
    options.max_iters = 200
    xbest, num_iters = cutting_plane_feas(omega, ellip, options)
    assert xbest is None
    assert num_iters == 2


def test_cutting_plane_optim(options):
    """Test cutting plane optimization."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim()
    options.max_iters = 200
    xbest, fbest, num_iters = cutting_plane_optim(omega, ellip, 0.0, options)
    assert xbest is not None
    assert fbest == pytest.approx(2.0)
    assert num_iters == 145


def test_cutting_plane_optim_no_soln(options):
    """Test cutting plane optimization with no solution."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptim()
    options.max_iters = 4
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 100.0, options)
    assert xbest is None
    assert num_iters == 0


def test_cutting_plane_optim_q(options):
    """Test cutting plane optimization with quantization."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptimQ()
    options.max_iters = 200
    xbest, fbest, num_iters = cutting_plane_optim_q(omega, ellip, 0.0, options)
    assert xbest is not None
    assert fbest == pytest.approx(2.0)
    assert num_iters == 145


def test_cutting_plane_optim_q_no_soln(options):
    """Test cutting plane optimization with quantization and no solution."""
    xinit = np.array([0.0, 0.0])
    ellip = Ell(10.0, xinit)
    omega = MyOracleOptimQ()
    options.max_iters = 20
    xbest, _, num_iters = cutting_plane_optim_q(omega, ellip, 100.0, options)
    assert xbest is None
    assert num_iters == 0


def test_bsearch(options):
    """Test binary search."""
    omega = MyOracleBS()
    options.tolerance = 1e-7
    gamma, num_iters = bsearch(omega, (-100.0, 100.0), options)
    assert gamma > 0.0
    assert gamma < 2e-7
    assert num_iters == 30


def test_bsearch_no_soln(options):
    """Test binary search with no solution."""
    omega = MyOracleBS()
    options.max_iters = 20
    gamma, num_iters = bsearch(omega, (-100.0, -50.0), options)
    assert gamma == -50.0
    assert num_iters == 20
