"""
Benchmark: CVXPY vs Ellipsoid method for the Chebyshev center problem.

Compares wall-clock run time of:
    1. CVXPY (CLARABEL interior-point solver, reference)
    2. ellalgo Ellipsoid method (cutting-plane)

across several problem sizes (n = design variables of the center,
m = number of random halfspaces).  Reports robust (median) solve time,
iterations, relative accuracy of the radius, and the solution quality
(max constraint violation, objective gap) achieved by both solvers so the
comparison is provably fair.

Both solvers are run with matched tolerances: CLARABEL's
``tol_gap_abs``/``tol_gap_rel``/``tol_feas`` are all set to the same
``TOLERANCE`` used for the ellipsoid ``Options.tolerance``.

Run it with:

    python benches/bench_chebyshev.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from _bench_util import Quality, quality_from_violations, time_alternating  # noqa: E402

from demo.chebyshev_center import ChebyshevOracle, generate_polyhedron  # noqa: E402
from ellalgo import Options, cutting_plane_optim  # noqa: E402
from ellalgo.ell import Ell  # noqa: E402

SIZES = [
    (5, 25),
    (10, 50),
    (15, 75),
    (20, 100),
    (30, 150),
]
REPEATS = 3
SEED = 0
TOLERANCE = 1e-10  # ellipsoid Options.tolerance, mirrored by CLARABEL


def solve_cvxpy(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve the Chebyshev center LP with CVXPY; return (center, radius)."""
    n = A.shape[1]
    x = cp.Variable(n)
    r = cp.Variable()
    constraints = [A[i] @ x + np.linalg.norm(A[i]) * r <= b[i] for i in range(len(b))]
    prob = cp.Problem(cp.Maximize(r), constraints)
    prob.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=TOLERANCE,
        tol_gap_rel=TOLERANCE,
        tol_feas=TOLERANCE,
    )
    return np.asarray(x.value), float(r.value)


def solve_ellipsoid(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Solve with the ellipsoid method and return (x, radius, iters)."""
    n = A.shape[1]
    omega = ChebyshevOracle(A, b)
    # Feasible region ⊂ ball of radius sqrt(n+1); scale kappa and the
    # iteration budget with n (ellipsoid method needs O(n^2) iterations).
    kappa = np.sqrt(n + 1) + 1.0
    space = Ell(kappa, np.zeros(n + 1))
    options = Options()
    options.tolerance = TOLERANCE
    options.max_iters = 500 * (n + 1) ** 2
    x_best, gamma, num_iters = cutting_plane_optim(omega, space, float("-inf"), options)
    return x_best, gamma, num_iters


def violation(A: np.ndarray, b: np.ndarray, x: np.ndarray, r: float) -> np.ndarray:
    """Constraint residuals ``aᵢᵀx + ‖aᵢ‖r - bᵢ`` (positive = violated)."""
    return A @ x + np.linalg.norm(A, axis=1) * r - b


def main() -> None:
    """Run the benchmark and print a comparison table."""
    print(
        f"{'n':>4} {'m':>5} {'cvxpy_med':>10} {'cvx_iqr':>8} {'ellip_med':>10} "
        f"{'ell_iqr':>8} {'iters':>6} {'ratio':>8} {'viol_cvx':>9} {'viol_ell':>9} "
        f"{'gap_r':>9}"
    )
    print("-" * 108)

    for n, m in SIZES:
        A, b = generate_polyhedron(n, m, SEED)
        (t_cvxpy, cvx_results), (t_ell, ell_results) = time_alternating(
            [lambda: solve_cvxpy(A, b), lambda: solve_ellipsoid(A, b)],
            repeat=REPEATS,
            warmup=1,
        )
        x_ref, r_ref = cvx_results[0]
        xc_em, r_em, num_iters = ell_results[0]

        q_cvx = quality_from_violations(violation(A, b, x_ref, r_ref), r_ref)
        if xc_em is None:
            q_ell = Quality(float("nan"), float("nan"))
        else:
            # xc_em = (x, r); the oracle's best radius is gamma (= r_em).
            q_ell = quality_from_violations(violation(A, b, xc_em[:-1], r_em), r_em)

        obj_gap = abs(q_cvx.objective - q_ell.objective)
        ratio = t_ell.median / t_cvxpy.median
        print(
            f"{n:>4} {m:>5} {t_cvxpy.median:>10.4f} {t_cvxpy.iqr:>8.4f} "
            f"{t_ell.median:>10.4f} {t_ell.iqr:>8.4f} {num_iters:>6} "
            f"{ratio:>7.1f}x {q_cvx.max_violation:>9.1e} {q_ell.max_violation:>9.1e} "
            f"{obj_gap:>9.1e}"
        )


if __name__ == "__main__":
    main()
