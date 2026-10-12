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

across several m.  Reports robust (median) solve time, iterations, and the
solution quality (max constraint violation, objective gap) of both solvers.
CLARABEL's ``tol_gap_abs``/``tol_gap_rel``/``tol_feas`` are all set to the
same ``TOLERANCE`` used for the ellipsoid ``Options.tolerance``.

Run it with:

    python benches/bench_lmi_evp.py
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
from _bench_util import Quality, quality_from_violations, time_alternating

from ellalgo import Ell, Options, OracleOptim, cutting_plane_optim
from ellalgo.oracles.lmi_oracle import LMIOracle

# Problem dimensions
N = 5  # LMI matrix size (n×n)
M_SIZES = [5, 8, 12, 16, 20]  # number of design variables xᵢ (total = m + 1)
REPEATS = 3
SEED = 0
TOLERANCE = 1e-10  # ellipsoid Options.tolerance, mirrored by CLARABEL


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


def solve_cvxpy(A0: np.ndarray, A_list: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """Solve the EVP with CVXPY and return (x, t*)."""
    m = len(A_list)
    x = cp.Variable(m)
    t = cp.Variable()
    A_expr = A0 + sum(x[i] * A_list[i] for i in range(m))
    constraints = [A_expr << t * np.eye(len(A0)), x >= -1.0, x <= 1.0]
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=TOLERANCE,
        tol_gap_rel=TOLERANCE,
        tol_feas=TOLERANCE,
    )
    return np.asarray(x.value), float(t.value)


def solve_ellipsoid(
    A0: np.ndarray, A_list: List[np.ndarray]
) -> Tuple[Optional[np.ndarray], float, int]:
    """Solve the EVP with the Ellipsoid method; return (x, t*, iters)."""
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
    options.tolerance = TOLERANCE
    options.max_iters = 20000
    x_best, gamma, num_iters = cutting_plane_optim(omega, space, float("inf"), options)
    return x_best, gamma, num_iters


def violation(
    A0: np.ndarray, A_list: List[np.ndarray], x: np.ndarray, t: float
) -> np.ndarray:
    """Constraint residuals for the EVP (positive = violated).

    Combines the LMI residual ``λ_max(A(x) - t·I)`` with the two-sided box
    residuals so both constraint families are covered.
    """
    m = len(A_list)
    A_expr = A0 + sum(x[i] * A_list[i] for i in range(m))
    eig = np.linalg.eigvalsh(A_expr - t * np.eye(len(A0)))
    box = np.abs(x) - 1.0
    return np.concatenate([eig, box])


def main() -> None:
    """Run the benchmark: solve each size with both methods, print comparison."""
    print(
        f"{'m':>3} {'vars':>4} {'cvxpy_med':>10} {'cvx_iqr':>8} {'ellip_med':>10} "
        f"{'ell_iqr':>8} {'iters':>6} {'ratio':>8} {'viol_cvx':>9} {'viol_ell':>9} "
        f"{'gap_t':>9}"
    )
    print("-" * 108)

    for m in M_SIZES:
        A0, A_list = generate_pencil(N, m, SEED)
        (t_cvxpy, cvx_results), (t_ell, ell_results) = time_alternating(
            [lambda: solve_cvxpy(A0, A_list), lambda: solve_ellipsoid(A0, A_list)],
            repeat=REPEATS,
            warmup=1,
        )
        x_ref, t_ref = cvx_results[0]
        xc_em, t_em, num_iters = ell_results[0]

        q_cvx = quality_from_violations(violation(A0, A_list, x_ref, t_ref), t_ref)
        if xc_em is None:
            q_ell = Quality(float("nan"), float("nan"))
        else:
            # xc_em = (x, t); the oracle's best objective is gamma (= t_em).
            q_ell = quality_from_violations(
                violation(A0, A_list, xc_em[:-1], t_em), t_em
            )

        obj_gap = abs(q_cvx.objective - q_ell.objective)
        ratio = t_ell.median / t_cvxpy.median
        print(
            f"{m:>3} {m + 1:>4} {t_cvxpy.median:>10.4f} {t_cvxpy.iqr:>8.4f} "
            f"{t_ell.median:>10.4f} {t_ell.iqr:>8.4f} {num_iters:>6} "
            f"{ratio:>7.1f}x {q_cvx.max_violation:>9.1e} {q_ell.max_violation:>9.1e} "
            f"{obj_gap:>9.1e}"
        )


if __name__ == "__main__":
    main()
