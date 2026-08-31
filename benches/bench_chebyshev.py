"""
Benchmark: CVXPY vs Ellipsoid method for the Chebyshev center problem.

Compares wall-clock run time of:
    1. CVXPY (CLARABEL interior-point solver, reference)
    2. ellalgo Ellipsoid method (cutting-plane)

across several problem sizes (n = design variables of the center,
m = number of random halfspaces).  Reports solve time, iterations and
relative accuracy of the radius.

Run it with:

    python benches/bench_chebyshev.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

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


def time_cvxpy(A: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Solve with CVXPY and return (wall time, radius)."""
    n = A.shape[1]
    x = cp.Variable(n)
    r = cp.Variable()
    constraints = [A[i] @ x + np.linalg.norm(A[i]) * r <= b[i] for i in range(len(b))]
    prob = cp.Problem(cp.Maximize(r), constraints)
    t0 = time.perf_counter()
    prob.solve()
    t1 = time.perf_counter()
    return t1 - t0, float(r.value)


def time_ellipsoid(A: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """Solve with the ellipsoid method and return (wall time, radius, iters)."""
    n = A.shape[1]
    omega = ChebyshevOracle(A, b)
    # Feasible region ⊂ ball of radius sqrt(n+1); scale kappa and the
    # iteration budget with n (ellipsoid method needs O(n^2) iterations).
    kappa = np.sqrt(n + 1) + 1.0
    space = Ell(kappa, np.zeros(n + 1))
    options = Options()
    options.tolerance = 1e-10
    options.max_iters = 500 * (n + 1) ** 2
    t0 = time.perf_counter()
    x_best, gamma, num_iters = cutting_plane_optim(omega, space, float("-inf"), options)
    t1 = time.perf_counter()
    return t1 - t0, gamma, num_iters


def main() -> None:
    """Run the benchmark and print a comparison table."""
    print(
        f"{'n':>4} {'m':>5} {'CVXPY (s)':>10} {'Ellipsoid (s)':>14} {'iters':>6} "
        f"{'ratio':>8} {'rel.err':>10}"
    )
    print("-" * 64)

    for n, m in SIZES:
        A, b = generate_polyhedron(n, m, SEED)
        t_cvxpy, r_ref = time_cvxpy(A, b)

        _, r_em, num_iters = time_ellipsoid(A, b)
        t_ell = 0.0
        for _ in range(REPEATS):
            t_ell += time_ellipsoid(A, b)[0]
        t_ell /= REPEATS

        rel_err = abs(r_em - r_ref) / r_ref
        ratio = t_ell / t_cvxpy
        print(
            f"{n:>4} {m:>5} {t_cvxpy:>10.4f} {t_ell:>14.4f} {num_iters:>6} "
            f"{ratio:>8.1f}x {rel_err:>10.1e}"
        )


if __name__ == "__main__":
    main()
