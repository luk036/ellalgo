"""
Stress tests for the ellalgo package.
"""

import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.oracles.lowpass_oracle import create_lowpass_case


def test_stress_lowpass_high_dimension() -> None:
    """
    Test with a high-dimensional problem for LowpassOracle.
    """
    n = 128  # number of variables
    p = create_lowpass_case(n)
    v = np.zeros(n)
    x, niter, feasible = cutting_plane_optim(p, Ell(1.0, v), 0.0)
    assert feasible


def test_stress_lowpass_many_iterations() -> None:
    """
    Test with a small tolerance to force many iterations for LowpassOracle.
    """
    n = 32
    p = create_lowpass_case(n)
    v = np.zeros(n)
    x, niter, feasible = cutting_plane_optim(p, Ell(1.0, v), 1e-12)
    assert feasible
