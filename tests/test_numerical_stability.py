"""
Numerical-stability tests, ported from ``ellalgo-rs/tests/numerical_stability.rs``.

These tests stress the ellipsoid search spaces (the direct ``Ell`` and the
LDL^T-based ``EllStable``) with poorly conditioned, badly scaled, and
ill-toleranced quadratic programs. The goal is not exact optima but *robust
convergence*: a feasible/improving point is returned and the tracked objective
stays finite instead of blowing up.

The Rust suite exercises ``Ell`` only; here every scenario is run against both
search spaces, and the opt-in ``Options(check_pd=True)`` positive-definiteness
check is enabled where it strengthens the verification.
"""

import math
from typing import Tuple, Type

import numpy as np
import pytest

from ellalgo import Ell, EllStable, Options, cutting_plane_optim

SPACES: Tuple[Type, ...] = (Ell, EllStable)


def _safe_gradient(g: np.ndarray) -> np.ndarray:
    """Return ``g`` unless it is the zero vector, in which case a fallback.

    ``Ell._omega`` rejects an all-zero gradient; at the exact optimum of a
    quadratic the gradient vanishes, so this guards the ill-conditioned sweeps
    against a spurious ``ValueError``.
    """
    return g if g.any() else np.array([1.0, 1.0])


class IllConditionedOracle:
    """Minimize ``f(x) = (1 + eps) x0^2 + (1 / eps) x1^2`` with ``eps = 1 / cond``.

    Mirrors the Rust ``IllConditionedOracle``: the curvature spread equals the
    requested condition number, making the level sets severely anisotropic.
    """

    def __init__(self, condition_number: float) -> None:
        self.condition_number = condition_number

    def assess_optim(
        self, xc: np.ndarray, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], "float | None"]:
        eps = 1.0 / self.condition_number
        gradient = _safe_gradient(
            np.array([2.0 * (1.0 + eps) * xc[0], 2.0 * (1.0 / eps) * xc[1]])
        )
        f = (1.0 + eps) * xc[0] ** 2 + (1.0 / eps) * xc[1] ** 2
        if f < gamma:
            return (gradient, f), f
        return (gradient, f), None


class QuadOracle:
    """Minimize ``f(x) = coeff0 * x0^2 + coeff1 * x1^2``.

    The Rust suite repeats this oracle three times (near-singular, tolerance,
    and far-start tests); a single parameterized helper keeps it in one place.
    """

    def __init__(self, coeff0: float = 1.0, coeff1: float = 1.0) -> None:
        self.coeff0 = coeff0
        self.coeff1 = coeff1

    def assess_optim(
        self, xc: np.ndarray, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], "float | None"]:
        gradient = _safe_gradient(
            np.array([2.0 * self.coeff0 * xc[0], 2.0 * self.coeff1 * xc[1]])
        )
        f = self.coeff0 * xc[0] ** 2 + self.coeff1 * xc[1] ** 2
        if f < gamma:
            return (gradient, f), f
        return (gradient, f), None


@pytest.mark.parametrize("space_cls", SPACES)
@pytest.mark.parametrize("condition_number", [1e3, 1e5, 1e7])
def test_ill_conditioned_quadratic(space_cls: Type, condition_number: float) -> None:
    """Sweep condition numbers [1e3, 1e5, 1e7]: must converge with finite gamma."""
    ellip = space_cls(10.0, np.array([1.0, 1.0]))
    omega = IllConditionedOracle(condition_number)
    options = Options(max_iters=2000, tolerance=1e-12, check_pd=True)

    xbest, gamma, _ = cutting_plane_optim(omega, ellip, math.inf, options)

    assert xbest is not None, f"Should converge with cond_num = {condition_number}"
    assert math.isfinite(
        gamma
    ), f"Final gamma should be finite for cond_num = {condition_number}"


@pytest.mark.parametrize("space_cls", SPACES)
def test_near_singular_initial_ellipsoid(space_cls: Type) -> None:
    """A near-singular start must still reach a reasonable objective."""
    ellip = space_cls(10.0, np.array([3.0, 3.0]))
    omega = QuadOracle()
    options = Options(max_iters=1000, tolerance=1e-10, check_pd=True)

    xbest, gamma, _ = cutting_plane_optim(omega, ellip, math.inf, options)

    assert xbest is not None, "Should handle near-singular initial ellipsoid"
    assert gamma < 10.0, f"Should converge to reasonable value: {gamma}"


@pytest.mark.parametrize("space_cls", SPACES)
@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_extreme_scale_values(space_cls: Type, scale: float) -> None:
    """Extreme coefficient scales [1e-6, 1e6] must stay finite and converge."""
    ellip = space_cls(10.0 * math.sqrt(abs(scale)), np.array([scale, scale]))
    omega = QuadOracle(coeff0=scale, coeff1=1.0)
    options = Options(max_iters=2000, tolerance=1e-10, check_pd=True)

    xbest, gamma, _ = cutting_plane_optim(omega, ellip, math.inf, options)

    assert xbest is not None, f"Should handle scale = {scale}"
    assert math.isfinite(gamma), f"gamma should be finite for scale = {scale}"


@pytest.mark.parametrize("space_cls", SPACES)
@pytest.mark.parametrize("tolerance", [1e-6, 1e-10, 1e-14])
def test_tolerance_sensitivity(space_cls: Type, tolerance: float) -> None:
    """Tolerances [1e-6, 1e-10, 1e-14] must all terminate successfully."""
    ellip = space_cls(10.0, np.array([3.0, 3.0]))
    omega = QuadOracle()
    options = Options(max_iters=2000, tolerance=tolerance, check_pd=True)

    xbest, gamma, _ = cutting_plane_optim(omega, ellip, math.inf, options)

    assert xbest is not None, f"Should converge with tolerance = {tolerance}"
    assert math.isfinite(gamma)


@pytest.mark.parametrize("space_cls", SPACES)
def test_numerical_precision(space_cls: Type) -> None:
    """Starting at (1000, -1000) the objective must strictly improve."""
    initial_point = np.array([1000.0, -1000.0])
    gamma_orig = initial_point[0] ** 2 + initial_point[1] ** 2
    ellip = space_cls(10.0, initial_point.copy())
    omega = QuadOracle()
    options = Options(max_iters=3000, tolerance=1e-12, check_pd=True)

    xbest, gamma, _ = cutting_plane_optim(omega, ellip, math.inf, options)

    assert xbest is not None, "Should converge from far starting point"
    assert gamma < gamma_orig, "gamma should improve from initial value"
