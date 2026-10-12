"""
Benchmark: CVXPY vs Ellipsoid method for FIR lowpass filter design.

The problem is the classic FIR lowpass filter design from Wu-Boyd-Vandenberghe
("FIR Filter Design via Spectral Factorization and Convex Optimization").
After spectral factorization the problem is convex in the autocorrelation
coefficients r (squared magnitude R(w) = A(w)r is linear):

    minimize    max R(w)                       for w in stopband
    subject to  lp_sq <= R(w) <= up_sq         for w in passband
                R(w) >= 0                      for all w

The design variables are the autocorrelation coefficients r ∈ R^ndim.
Here ndim = 32, so the problem has 32 design variables — "moderate".
The frequency grid uses the rule-of-thumb mdim = 15*ndim points.

The benchmark compares:
    1. CVXPY (CLARABEL, reference) — same spectral-factorization LP
    2. ellalgo Ellipsoid method (LowpassOracle + cutting_plane_optim)

across several filter lengths.  Reports robust (median) solve time,
iterations, the achieved stopband attenuation in dB, and the solution
quality (max constraint violation, objective gap) of both solvers.
CLARABEL's ``tol_gap_abs``/``tol_gap_rel``/``tol_feas`` are all set to the
same ``TOLERANCE`` used for the ellipsoid ``Options.tolerance``.

Run it with:

    python benches/bench_lowpass_cvxpy.py
"""

from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import numpy as np
from _bench_util import Quality, quality_from_violations, time_alternating

from ellalgo import Ell, Options, cutting_plane_optim
from ellalgo.oracles.lowpass_oracle import LowpassOracle

# Filter specs (mirror create_lowpass_case)
NPASS = 0.12  # passband edge (fraction of pi)
NSTOP = 0.20  # stopband edge (fraction of pi)
DELTA0_WPASS = 0.025  # passband ripple tolerance
DELTA0_WSTOP = 0.125  # stopband attenuation tolerance
# Filter lengths (design variables). n=16 is omitted: with these specs a
# length-16 filter cannot meet the stopband bound, so the problem is infeasible.
NDIM_SIZES = [24, 32, 48, 64, 80]
REPEATS = 3
SEED = 0
TOLERANCE = 1e-14  # ellipsoid Options.tolerance, mirrored by CLARABEL


def make_lowpass_bounds() -> Tuple[float, float, float]:
    """Compute the squared-magnitude bounds (lp_sq, up_sq, sp_sq)."""
    delta1 = 20 * np.log10(1 + DELTA0_WPASS)
    delta2 = 20 * np.log10(DELTA0_WSTOP)
    low_pass = pow(10, -delta1 / 20)
    up_pass = pow(10, +delta1 / 20)
    stop_pass = pow(10, +delta2 / 20)
    return low_pass * low_pass, up_pass * up_pass, stop_pass * stop_pass


def build_spectrum(ndim: int) -> np.ndarray:
    """Build the power-spectrum cosine matrix A (mdim x ndim).

    Rows are frequency points; A[k, :] = [1, 2cos(w_k), 2cos(2w_k), ...].
    """
    mdim = 15 * ndim
    w = np.linspace(0, np.pi, mdim)
    temp = 2 * np.cos(np.outer(w, np.arange(1, ndim)))
    return np.concatenate((np.ones((mdim, 1)), temp), axis=1)


def build_constraint_data(
    ndim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (A, Ap, As, lp_sq, up_sq) for the lowpass LP."""
    lp_sq, up_sq, _ = make_lowpass_bounds()
    A = build_spectrum(ndim)
    w = np.linspace(0, np.pi, 15 * ndim)
    ipass = w <= NPASS * np.pi
    istop = w >= NSTOP * np.pi
    return A, A[ipass, :], A[istop, :], lp_sq, up_sq


def solve_cvxpy(ndim: int) -> Tuple[np.ndarray, float]:
    """Solve the FIR lowpass LP with CVXPY; return (r, stopband max)."""
    A, Ap, As, lp_sq, up_sq = build_constraint_data(ndim)
    r = cp.Variable(ndim)
    t = cp.Variable()
    constraints = [
        Ap @ r <= up_sq,
        Ap @ r >= lp_sq,
        As @ r <= t,
        -As @ r <= t,
        A @ r >= 0,
    ]
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=TOLERANCE,
        tol_gap_rel=TOLERANCE,
        tol_feas=TOLERANCE,
    )
    return np.asarray(r.value), float(t.value)


def solve_ellipsoid(ndim: int) -> Tuple[np.ndarray, float, int]:
    """Solve the FIR lowpass problem with the Ellipsoid method.

    Returns (r, stopband max, iterations).
    """
    lp_sq, up_sq, sp_sq = make_lowpass_bounds()
    omega = LowpassOracle(ndim, NPASS, NSTOP, lp_sq, up_sq, sp_sq)
    r0 = np.zeros(ndim)
    ellip = Ell(40.0, r0)
    ellip.helper.use_parallel_cut = True
    options = Options()
    options.tolerance = TOLERANCE
    options.max_iters = 50000
    x_best, gamma, num_iters = cutting_plane_optim(omega, ellip, omega.sp_sq, options)
    return x_best, gamma, num_iters


def violation(ndim: int, r: np.ndarray, t: float) -> np.ndarray:
    """Constraint residuals for the lowpass LP (positive = violated)."""
    A, Ap, As, lp_sq, up_sq = build_constraint_data(ndim)
    return np.concatenate(
        [
            Ap @ r - up_sq,
            lp_sq - Ap @ r,
            As @ r - t,
            -As @ r - t,
            -(A @ r),
        ]
    )


def main() -> None:
    """Run the benchmark: solve each filter length with both methods, compare."""
    print(
        f"{'n':>3} {'cvxpy_med':>10} {'cvx_iqr':>8} {'ellip_med':>10} {'ell_iqr':>8} "
        f"{'iters':>6} {'ratio':>8} {'viol_cvx':>9} {'viol_ell':>9} {'cvx_dB':>8} "
        f"{'ell_dB':>8} {'gap_dB':>8}"
    )
    print("-" * 116)

    for ndim in NDIM_SIZES:
        (t_cvxpy, cvx_results), (t_ell, ell_results) = time_alternating(
            [lambda: solve_cvxpy(ndim), lambda: solve_ellipsoid(ndim)],
            repeat=REPEATS,
            warmup=1,
        )
        r_ref, t_ref = cvx_results[0]
        r_em, t_em, num_iters = ell_results[0]

        q_cvx = quality_from_violations(violation(ndim, r_ref, t_ref), t_ref)
        if r_em is None:
            q_ell = Quality(float("nan"), float("nan"))
        else:
            q_ell = quality_from_violations(violation(ndim, r_em, t_em), t_em)

        db_cvxpy = 20 * np.log10(np.sqrt(q_cvx.objective))
        db_em = 20 * np.log10(np.sqrt(q_ell.objective))
        gap_db = abs(db_cvxpy - db_em)
        ratio = t_ell.median / t_cvxpy.median
        print(
            f"{ndim:>3} {t_cvxpy.median:>10.4f} {t_cvxpy.iqr:>8.4f} "
            f"{t_ell.median:>10.4f} {t_ell.iqr:>8.4f} {num_iters:>6} "
            f"{ratio:>7.1f}x {q_cvx.max_violation:>9.1e} {q_ell.max_violation:>9.1e} "
            f"{db_cvxpy:>8.2f} {db_em:>8.2f} {gap_db:>8.2f}"
        )


if __name__ == "__main__":
    main()
