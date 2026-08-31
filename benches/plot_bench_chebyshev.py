"""
Plot run-time comparison: CVXPY vs Ellipsoid method (n = 2 .. 10).

Benchmarks the Chebyshev center problem at every dimension n in
[2, 10] (design variables = n + 1, m = 5n random halfspaces), then
plots wall-clock time vs n on a log scale.  Saves the figure to
``benches/bench_chebyshev.svg``.

Run it with:

    python benches/plot_bench_chebyshev.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from demo.chebyshev_center import ChebyshevOracle, generate_polyhedron  # noqa: E402
from ellalgo import Options, cutting_plane_optim  # noqa: E402
from ellalgo.ell import Ell  # noqa: E402

N_RANGE = range(2, 11)
REPEATS = 5
SEED = 0


def bench_one(n: int, m: int) -> Tuple[float, float, int]:
    """Benchmark a single problem size; return (cvxpy_s, ell_s, iters)."""
    A, b = generate_polyhedron(n, m, SEED)

    # CVXPY
    x = cp.Variable(n)
    r = cp.Variable()
    constraints = [A[i] @ x + np.linalg.norm(A[i]) * r <= b[i] for i in range(len(b))]
    prob = cp.Problem(cp.Maximize(r), constraints)
    t0 = time.perf_counter()
    prob.solve()
    t_cvxpy = time.perf_counter() - t0

    # Ellipsoid method (best of REPEATS)
    t_ell = float("inf")
    iters = 0
    for _ in range(REPEATS):
        omega = ChebyshevOracle(A, b)
        kappa = np.sqrt(n + 1) + 1.0
        space = Ell(kappa, np.zeros(n + 1))
        options = Options()
        options.tolerance = 1e-10
        options.max_iters = 500 * (n + 1) ** 2
        t0 = time.perf_counter()
        _, _, num_iters = cutting_plane_optim(omega, space, float("-inf"), options)
        t_ell = min(t_ell, time.perf_counter() - t0)
        iters = num_iters

    return t_cvxpy, t_ell, iters


def main() -> None:
    """Benchmark all sizes, plot the comparison, save the figure."""
    ns: List[int] = []
    t_cvxpy_list: List[float] = []
    t_ell_list: List[float] = []

    for n in N_RANGE:
        m = 5 * n
        t_cvxpy, t_ell, iters = bench_one(n, m)
        ns.append(n)
        t_cvxpy_list.append(t_cvxpy)
        t_ell_list.append(t_ell)
        print(
            f"n={n:>2}  m={m:>3}  CVXPY={t_cvxpy:.4f}s  Ellipsoid={t_ell:.4f}s  iters={iters}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, t_cvxpy_list, "o-", label="CVXPY (CLARABEL)", color="#1565C0")
    ax.plot(ns, t_ell_list, "s-", label="Ellipsoid method (ellalgo)", color="#E65100")
    ax.set_yscale("log")
    ax.set_xlabel("dimension n  (design variables = n + 1)")
    ax.set_ylabel("wall-clock time (s, log scale)")
    ax.set_title("Run-time comparison: Chebyshev center problem")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "bench_chebyshev.svg"
    fig.savefig(out, format="svg")
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
