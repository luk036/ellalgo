"""Verification helpers for separation oracles.

The ellipsoid method is only as sound as the cut its oracle returns: an invalid
cut or a wrong (sub)gradient yields a wrong answer with no signal in the
iteration count. These helpers make oracle correctness testable by checking the
defining inequalities directly, at known points, independent of the solver.

The functions return ``bool``; the matching ``verify_*`` wrappers raise
``AssertionError`` with a diagnostic message so they can be used as drop-in
assertions inside a test suite.
"""

from typing import Callable, Optional, Sequence

import numpy as np

from .ell_typing import ArrayType, Cut


def check_cut_valid(
    cut: Cut,
    x_center: ArrayType,
    feasible_points: Sequence[ArrayType],
    tol: float = 1e-8,
) -> bool:
    r"""Check that a single cut does not separate any known feasible point.

    A separating cut :math:`g^T(x - x_0) + \beta \le 0` is valid only if it
    holds at every point of the feasible set. A parallel cut
    :math:`(g, (\beta_0, \beta_1))` with :math:`\beta_0 \le \beta_1` encodes the
    two half-spaces :math:`g^T(x - x_0) + \beta_0 \le 0` and
    :math:`g^T(x - x_0) + \beta_1 \ge 0`, i.e. the strip
    :math:`-\beta_1 \le g^T(x - x_0) \le -\beta_0` (the convention used by
    :class:`~ellalgo.oracles.lowpass_oracle.LowpassOracle`), so both bounds are
    checked.

    Args:
        cut: The ``(g, beta)`` pair returned by the oracle.
        x_center: The query point ``x_0`` at which the cut was generated.
        feasible_points: Points known to lie in the feasible set.
        tol: Absolute tolerance on the inequality.

    Returns:
        True when every feasible point satisfies the cut within ``tol``.
    """
    g, beta = cut
    g = np.asarray(g, dtype=float)
    if not feasible_points:
        return True
    if isinstance(beta, (int, float)):
        for x in feasible_points:
            if g.dot(np.asarray(x, dtype=float) - x_center) + beta > tol:
                return False
        return True
    b0 = beta[0]
    b1 = beta[1]
    assert b0 is not None and b1 is not None
    lo, hi = -float(b1), -float(b0)
    for x in feasible_points:
        v = g.dot(np.asarray(x, dtype=float) - x_center)
        if v < lo - tol or v > hi + tol:
            return False
    return True


def verify_cut_valid(
    cut: Cut,
    x_center: ArrayType,
    feasible_points: Sequence[ArrayType],
    tol: float = 1e-8,
) -> None:
    """Raise :class:`AssertionError` unless :func:`check_cut_valid` holds."""
    if not check_cut_valid(cut, x_center, feasible_points, tol):
        raise AssertionError(
            f"cut {cut!r} separates a known feasible point at x_center={x_center!r}"
        )


def check_subgradient(
    func: Callable[[np.ndarray], float],
    x: ArrayType,
    grad: ArrayType,
    step: float = 1e-6,
    tol: float = 1e-4,
    directions: Optional[Sequence[ArrayType]] = None,
) -> bool:
    r"""Check the subgradient inequality by finite differences.

    For convex :math:`f`, a valid subgradient at ``x`` satisfies
    :math:`f(x + t d) \ge f(x) + t\, g^T d` for every direction ``d`` and small
    :math:`t > 0`. The check evaluates this along the coordinate directions and
    (optionally) user-supplied directions.

    Args:
        func: The convex function ``f``.
        x: Point at which the subgradient is claimed.
        grad: Claimed subgradient ``g``.
        step: Perturbation size ``t``.
        tol: Absolute tolerance on the inequality.
        directions: Extra directions to probe; defaults to ±coordinate axes.

    Returns:
        True when the inequality holds in every probed direction.
    """
    xs = np.asarray(x, dtype=float)
    gs = np.asarray(grad, dtype=float)
    f0 = float(func(xs))
    if directions is None:
        dirs = []
        for i in range(xs.size):
            e = np.zeros_like(xs)
            e[i] = 1.0
            dirs.append(e)
            dirs.append(-e)
    else:
        dirs = [np.asarray(d, dtype=float) for d in directions]
    for d in dirs:
        lhs = float(func(xs + step * d))
        rhs = f0 + step * gs.dot(d)
        if lhs < rhs - tol:
            return False
    return True


def verify_subgradient(
    func: Callable[[np.ndarray], float],
    x: ArrayType,
    grad: ArrayType,
    step: float = 1e-6,
    tol: float = 1e-4,
    directions: Optional[Sequence[ArrayType]] = None,
) -> None:
    """Raise :class:`AssertionError` unless :func:`check_subgradient` holds."""
    if not check_subgradient(func, x, grad, step, tol, directions):
        raise AssertionError(f"gradient {grad!r} is not a subgradient of f at x={x!r}")


def check_lmi_witness(mat_f: np.ndarray, witness: ArrayType, tol: float = 0.0) -> bool:
    r"""Check that a witness vector certifies a matrix is not positive definite.

    The certificate is :math:`v^T F v < 0` (strictly negative quadratic form),
    which proves ``F`` is not positive semidefinite.

    Args:
        mat_f: The symmetric matrix ``F`` being tested.
        witness: The claimed witness vector ``v``.
        tol: Require :math:`v^T F v < -tol` (0 means strictly negative).

    Returns:
        True when the witness is a valid negative-curvature certificate.
    """
    v = np.asarray(witness, dtype=float)
    F = np.asarray(mat_f, dtype=float)
    return float(v.dot(F).dot(v)) < -tol


def verify_lmi_witness(mat_f: np.ndarray, witness: ArrayType, tol: float = 0.0) -> None:
    """Raise :class:`AssertionError` unless :func:`check_lmi_witness` holds."""
    if not check_lmi_witness(mat_f, witness, tol):
        v = np.asarray(witness, dtype=float)
        raise AssertionError(
            f"witness {witness!r} is not a negative-curvature certificate for F: "
            f"v^T F v = {float(v.dot(np.asarray(mat_f, dtype=float)).dot(v)):.3e}"
        )
