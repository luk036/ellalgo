"""
Chebyshev Center of a Polyhedron — solved with the Ellipsoid Method
===================================================================

This demo solves a classical convex optimization problem — finding the
largest Euclidean ball that fits inside a polyhedron — using the
Ellipsoid method from ``ellalgo``, and cross-checks the result against
the reference solution computed with CVXPY.

Problem
-------

Given a polyhedron defined by linear inequalities,

    P = { u ∈ R^n  :  A u ≤ b },

find the largest ball B(x, r) = { u : ‖u − x‖ ≤ r } contained in P.
This is the *Chebyshev center* problem (Boyd & Vandenberghe, §8.5.1):

    maximize    r
    subject to  aᵢᵀx + ‖aᵢ‖₂ r ≤ bᵢ,   i = 1, …, m,

with design variables ``(x, r) ∈ R^{n+1}``.  Here ``n = 10``, so the
problem has **11 design variables** and ``m = 50`` affine constraints.

Why the Ellipsoid method?
-------------------------

Every constraint is *affine* in ``(x, r)``, so its gradient
``(aᵢ, ‖aᵢ‖₂)`` is a constant — each violated constraint yields a
perfect cutting plane.  The objective is linear, so the optimality cut
is equally simple.  This makes the problem an ideal showcase for the
cutting-plane (Ellipsoid) method.

The demo:

1. generates a random bounded polyhedron,
2. formulates the problem in CVXPY and solves it (reference),
3. solves the same problem with the Ellipsoid method (ellalgo),
4. prints a comparison of the two solutions.

Run it with:

    python demo/chebyshev_center.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from ellalgo import Ell, Options, cutting_plane_optim
from ellalgo.ell_typing import OracleOptim

# Problem dimensions
N = 10  # dimension of the ball center (→ 11 design variables)
M = 50  # number of random halfspaces (plus 2N box constraints)
SEED = 0


def generate_polyhedron(n: int, m: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a random bounded polyhedron P = {x : A x ≤ b}.

    The polyhedron is the intersection of the box [-1, 1]^n (which makes
    it bounded) with ``m`` random halfspaces whose offsets are positive,
    so the origin is strictly feasible and the Chebyshev radius is > 0.

    Args:
        n: ambient dimension of the polyhedron.
        m: number of random halfspaces.
        seed: random seed for reproducibility.

    Returns:
        A tuple ``(A, b)`` of the inequality data, with one row of ``A``
        per constraint.
    """
    rng = np.random.default_rng(seed)
    # Box constraints: -1 ≤ x_i ≤ 1
    A_box = np.vstack([np.eye(n), -np.eye(n)])
    b_box = np.ones(2 * n)
    # Random halfspaces: unit normals aᵢ, offsets in (0.3, 0.8)
    A_rand = rng.standard_normal((m, n))
    A_rand /= np.linalg.norm(A_rand, axis=1, keepdims=True)
    b_rand = rng.uniform(0.3, 0.8, size=m)
    A = np.vstack([A_box, A_rand])
    b = np.concatenate([b_box, b_rand])
    return A, b


class ChebyshevOracle(OracleOptim):
    """Optimization oracle for the Chebyshev center problem.

    The search point is ``xc = (x, r)``, where ``x`` is the ball center
    and ``r`` its radius.  The oracle scans the halfspaces in a
    round-robin fashion and returns a cut for the first violated
    constraint; when the point is feasible it returns the optimality cut
    that drives the radius upward.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self._A = A
        self._b = b
        self._norms = np.linalg.norm(A, axis=1)
        self._num_constraints = len(b)
        self.idx = -1  # round-robin counter

    def assess_optim(
        self, xc: np.ndarray, gamma: float
    ) -> Tuple[Tuple[np.ndarray, float], Optional[float]]:
        """Assess feasibility and optimality at ``xc``.

        Args:
            xc: candidate point ``(x, r)``.
            gamma: best-so-far objective value (radius).

        Returns:
            A pair ``(cut, gamma_new)``: the cutting plane and, when the
            objective improves, the new objective value.
        """
        x, r = xc[:-1], xc[-1]

        # --- Feasibility: aᵢᵀx + ‖aᵢ‖ r ≤ bᵢ (round robin) ---
        for _ in range(self._num_constraints):
            self.idx = (self.idx + 1) % self._num_constraints
            i = self.idx
            if (fj := self._A[i] @ x + self._norms[i] * r - self._b[i]) > 0.0:
                g = np.concatenate([self._A[i], [self._norms[i]]])
                return (g, fj), None

        # --- Optimality: maximize f0 = r ---
        f0 = r
        if (fj := gamma - f0) > 0.0:
            # Cut toward improving the objective: -(r - gamma) ≤ 0
            g = np.zeros(len(xc))
            g[-1] = -1.0
            return (g, fj), None
        # Improved: central cut on the objective
        g = np.zeros(len(xc))
        g[-1] = -1.0
        return (g, 0.0), f0


def solve_cvxpy(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve the Chebyshev center problem with CVXPY (reference).

    Args:
        A: inequality matrix.
        b: inequality offsets.

    Returns:
        The optimal ``(x, r)``.
    """
    n = A.shape[1]
    x = cp.Variable(n)
    r = cp.Variable()
    constraints = [A[i] @ x + np.linalg.norm(A[i]) * r <= b[i] for i in range(len(b))]
    prob = cp.Problem(cp.Maximize(r), constraints)
    prob.solve()
    return np.asarray(x.value), float(r.value)


def solve_ellipsoid(
    A: np.ndarray, b: np.ndarray
) -> Tuple[Optional[np.ndarray], float, int]:
    """Solve the Chebyshev center problem with the Ellipsoid method.

    Args:
        A: inequality matrix.
        b: inequality offsets.

    Returns:
        The best ``(x, r)`` found plus the iteration count.
    """
    n = A.shape[1]
    omega = ChebyshevOracle(A, b)
    # The feasible region lies in [-1, 1]^n × [0, 1] ⊂ ball of radius √(n+1).
    space = Ell(5.0, np.zeros(n + 1))
    options = Options()
    options.tolerance = 1e-10
    options.max_iters = 5000
    x_best, gamma, num_iters = cutting_plane_optim(omega, space, float("-inf"), options)
    return x_best, gamma, num_iters


def main() -> None:
    """Run the demo: generate the problem, solve it both ways, compare."""
    A, b = generate_polyhedron(N, M, SEED)
    num_vars = N + 1
    num_constr = len(b)

    print("=" * 64)
    print("Chebyshev center of a polyhedron - Ellipsoid method demo")
    print("=" * 64)
    print(f"design variables (x, r) : {num_vars}  (x in R^{N})")
    print(f"linear constraints      : {num_constr}")
    print()

    # Reference solution (CVXPY)
    x_ref, r_ref = solve_cvxpy(A, b)
    print(f"CVXPY reference  : r* = {r_ref:12.6f}")

    # Ellipsoid method
    x_best, r_em, num_iters = solve_ellipsoid(A, b)
    if x_best is None:
        print(
            "Ellipsoid method: FAILED (no solution found — enlarge the initial ellipsoid)"
        )
        return
    print(f"Ellipsoid method : r* = {r_em:12.6f}  ({num_iters} iterations)")

    # Feasibility check of the ellipsoid-method solution
    slack = A @ x_best[:-1] + np.linalg.norm(A, axis=1) * r_em - b
    print(f"max constraint violation : {np.max(slack):.3e}")

    # Comparison
    print()
    print("-" * 64)
    print("Comparison")
    print("-" * 64)
    print(
        f"radius  : CVXPY = {r_ref:.8f}  |  Ellipsoid = {r_em:.8f}  |  rel. err = "
        f"{abs(r_em - r_ref) / r_ref:.2e}"
    )
    print(f"center  : ||x_cvxpy - x_ell|| = {np.linalg.norm(x_ref - x_best[:-1]):.2e}")
    print()
    print("center (Ellipsoid):", np.round(x_best[:-1], 4))
    print("center (CVXPY)    :", np.round(x_ref, 4))


if __name__ == "__main__":
    main()
