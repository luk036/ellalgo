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

across several filter lengths.  Reports solve time, iterations, and the
achieved stopband attenuation in dB.

Run it with:

    python benches/bench_lowpass_cvxpy.py
"""

from __future__ import annotations

import time
from typing import Tuple

import cvxpy as cp
import numpy as np

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


def time_cvxpy(ndim: int) -> Tuple[float, float]:
    """Solve the FIR lowpass LP with CVXPY; return (wall time, stopband max)."""
    lp_sq, up_sq, sp_sq = make_lowpass_bounds()
    A = build_spectrum(ndim)
    w = np.linspace(0, np.pi, 15 * ndim)
    ipass = w <= NPASS * np.pi
    istop = w >= NSTOP * np.pi
    Ap, As = A[ipass, :], A[istop, :]

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
    t0 = time.perf_counter()
    prob.solve()
    t1 = time.perf_counter()
    return t1 - t0, float(t.value)


def time_ellipsoid(ndim: int) -> Tuple[float, float, int]:
    """Solve the FIR lowpass problem with the Ellipsoid method.

    Returns (wall time, stopband max, iterations).
    """
    lp_sq, up_sq, sp_sq = make_lowpass_bounds()
    omega = LowpassOracle(ndim, NPASS, NSTOP, lp_sq, up_sq, sp_sq)
    r0 = np.zeros(ndim)
    ellip = Ell(40.0, r0)
    ellip.helper.use_parallel_cut = True
    options = Options()
    options.tolerance = 1e-14
    options.max_iters = 50000
    t0 = time.perf_counter()
    x_best, gamma, num_iters = cutting_plane_optim(omega, ellip, omega.sp_sq, options)
    t1 = time.perf_counter()
    if x_best is None:
        return t1 - t0, float("nan"), num_iters
    return t1 - t0, gamma, num_iters


def main() -> None:
    """Run the benchmark: solve each filter length with both methods, compare."""
    print(
        f"{'n':>3} {'CVXPY (s)':>10} {'Ellipsoid (s)':>14} {'iters':>6} "
        f"{'ratio':>8} {'cvxpy_dB':>9} {'ellip_dB':>9}"
    )
    print("-" * 70)

    for ndim in NDIM_SIZES:
        t_cvxpy, t_ref = time_cvxpy(ndim)
        t_ell, t_em, num_iters = time_ellipsoid(ndim)
        for _ in range(REPEATS - 1):
            t_ell = min(t_ell, time_ellipsoid(ndim)[0])

        db_cvxpy = 20 * np.log10(np.sqrt(t_ref))
        db_em = 20 * np.log10(np.sqrt(t_em))
        ratio = t_ell / t_cvxpy
        print(
            f"{ndim:>3} {t_cvxpy:>10.4f} {t_ell:>14.4f} {num_iters:>6} "
            f"{ratio:>8.1f}x {db_cvxpy:>9.2f} {db_em:>9.2f}"
        )


if __name__ == "__main__":
    main()
