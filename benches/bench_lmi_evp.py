"""
Benchmark: CVXPY vs Ellipsoid method for a min-eigenvalue LMI (SDP) problem.

The problem is the classic eigenvalue problem (EVP) from semidefinite
programming: minimize the largest eigenvalue of an affine matrix pencil,

    minimize    t
    subject to  t·I - (A0 + Σᵢ xᵢ Aᵢ) ⪰ 0,   -1 ≤ xᵢ ≤ 1,

with design variables (x, t) ∈ R^{m+1}.  Here m = 5..20, so the problem
has 6..21 design variables — "moderate".  The LMI is affine in (x, t),
so every violated constraint yields a perfect cutting plane from an
LDL^T witness vector (LMIOracle).

The benchmark compares:
    1. CVXPY (CLARABEL interior-point solver, reference)
    2. ellalgo Ellipsoid method (LMIOracle + cutting_plane_optim)

across several m.  Reports solve time, iterations, and relative accuracy
of the optimal eigenvalue bound t*.

Run it with:

    python benches/bench_lmi_evp.py
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np

from ellalgo import Ell, Options, OracleOptim, cutting_plane_optim
from ellalgo.oracles.lmi_oracle import LMIOracle

# Problem dimensions
N = 5  # LMI matrix size (n×n)
M_SIZES = [5, 8, 12, 16, 20]  # number of design variables xᵢ (total = m + 1)
REPEATS = 3
SEED = 0


def make_symmetric(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate a random symmetric n×n matrix."""
    mat = rng.standard_normal((n, n))
    return (mat + mat.T) / 2.0


def generate_pencil(n: int, m: int, seed: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate a random affine matrix pencil A0 + Σ xᵢAᵢ.

    Args:
        n: LMI matrix dimension.
        m: number of design variables.
        seed: random seed for reproducibility.

    Returns:
        A tuple (A0, A_list) with one symmetric n×n matrix per variable.
    """
    rng = np.random.default_rng(seed)
    A0 = make_symmetric(rng, n)
    A_list = [make_symmetric(rng, n) for _ in range(m)]
    return A0, A_list


class EigenOracle(OracleOptim):
    """Optimization oracle for the min-eigenvalue EVP.

    The search point is xc = (x, t) ∈ R^{m+1}.  The oracle checks the box
    constraints, then the LMI t·I − A(x) ⪰ 0 via LMIOracle (LDL^T witness
    cut), then the linear objective t.
    """

    def __init__(self, A0: np.ndarray, A_list: List[np.ndarray]) -> None:
        self._m = len(A_list)
        self._c = np.zeros(self._m + 1)
        self._c[-1] = 1.0  # objective: minimize t
        # LMI: t·I − A0 − Σ xᵢAᵢ ⪰ 0  ⇔  B − Σ Fₖ·xcₖ ⪰ 0 with
        #   B = −A0,  Fᵢ = Aᵢ (i=1..m),  F_{m+1} = −I
        mat_f = A_list + [-np.eye(len(A0))]
        self._lmi = LMIOracle(mat_f, -A0)

    def assess_optim(
        self, xc: np.ndarray, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], Optional[float]]:
        """Assess feasibility and optimality at ``xc``.

        Args:
            xc: candidate point (x, t).
            gamma: best-so-far objective value (upper bound on t*).

        Returns:
            A pair (cut, gamma_new): the cutting plane and, when the
            objective improves, the new objective value.
        """
        for i in range(self._m):  # box constraint: -1 ≤ xᵢ ≤ 1
            if (fj := xc[i] - 1.0) > 0.0:
                g = np.zeros(self._m + 1)
                g[i] = 1.0
                return (g, fj), None
            if (fj := -1.0 - xc[i]) > 0.0:
                g = np.zeros(self._m + 1)
                g[i] = -1.0
                return (g, fj), None
        if cut := self._lmi.assess_feas(xc):
            return cut, None  # LMI violated -> deep cut
        f0 = self._c.dot(xc)  # objective: minimize t
        if (fj := f0 - gamma) > 0.0:
            return (self._c, fj), None  # deep objective cut
        return (self._c, 0.0), f0  # improved -> central cut


def time_cvxpy(A0: np.ndarray, A_list: List[np.ndarray]) -> Tuple[float, float]:
    """Solve the EVP with CVXPY and return (wall time, t*)."""
    m = len(A_list)
    x = cp.Variable(m)
    t = cp.Variable()
    A_expr = A0 + sum(x[i] * A_list[i] for i in range(m))
    constraints = [A_expr << t * np.eye(len(A0)), x >= -1.0, x <= 1.0]
    prob = cp.Problem(cp.Minimize(t), constraints)
    t0 = time.perf_counter()
    prob.solve()
    t1 = time.perf_counter()
    return t1 - t0, float(t.value)


def time_ellipsoid(
    A0: np.ndarray, A_list: List[np.ndarray]
) -> Tuple[float, float, int]:
    """Solve the EVP with the Ellipsoid method; return (time, t*, iters)."""
    m = len(A_list)
    omega = EigenOracle(A0, A_list)
    # Feasible region: x ∈ [-1,1]^m and t bounded by ‖A(x)‖₂; bound t from
    # t·I ⪰ A(x) ⇒ t ≤ ‖A0‖₂ + √m·max‖Aᵢ‖₂.  Scale kappa to contain it.
    bound_t = np.linalg.norm(A0, 2) + np.sqrt(m) * max(
        np.linalg.norm(A, 2) for A in A_list
    )
    kappa = np.sqrt(m + bound_t**2) + 1.0
    space = Ell(kappa, np.zeros(m + 1))
    options = Options()
    options.tolerance = 1e-10
    options.max_iters = 20000
    t0 = time.perf_counter()
    x_best, gamma, num_iters = cutting_plane_optim(omega, space, float("inf"), options)
    t1 = time.perf_counter()
    if x_best is None:
        return t1 - t0, float("nan"), num_iters
    return t1 - t0, gamma, num_iters


def main() -> None:
    """Run the benchmark: solve each size with both methods, print comparison."""
    print(
        f"{'m':>3} {'vars':>4} {'CVXPY (s)':>10} {'Ellipsoid (s)':>14} {'iters':>6} "
        f"{'ratio':>8} {'rel.err':>10}"
    )
    print("-" * 64)

    for m in M_SIZES:
        A0, A_list = generate_pencil(N, m, SEED)
        t_cvxpy, t_ref = time_cvxpy(A0, A_list)

        t_ell, t_em, num_iters = time_ellipsoid(A0, A_list)
        for _ in range(REPEATS - 1):
            t_ell = min(t_ell, time_ellipsoid(A0, A_list)[0])

        rel_err = abs(t_em - t_ref) / abs(t_ref)
        ratio = t_ell / t_cvxpy
        print(
            f"{m:>3} {m + 1:>4} {t_cvxpy:>10.4f} {t_ell:>14.4f} {num_iters:>6} "
            f"{ratio:>8.1f}x {rel_err:>10.1e}"
        )


if __name__ == "__main__":
    main()
