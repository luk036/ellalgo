"""
Shared helpers for the CVXPY-vs-ellipsoid benchmarks.

Two concerns are handled here so every benchmark script measures the two
solvers the same way:

1. **Robust wall-clock timing.**  A single ``time.perf_counter`` sample is
   dominated by scheduler noise and (for CVXPY) one-off compilation.  The
   helpers below run each callable several times and report the *median*
   wall-clock time together with the interquartile range and the minimum,
   which is a far more stable and reproducible comparison.  The alternating
   driver interleaves the ellipsoid and CVXPY runs so that slow drift in
   machine load does not systematically favour either solver.

2. **Solution quality.**  Timing is only meaningful if both solvers reached
   a comparable solution.  ``quality_from_violations`` turns a vector of
   constraint residuals (positive = violated) into the maximum constraint
   violation alongside the objective value, so each script can print the
   quality of *both* solutions and prove they converged comparably.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, NamedTuple, Sequence, Tuple

import numpy as np


class TimingStats(NamedTuple):
    """Robust statistics of repeated wall-clock measurements (seconds).

    Attributes:
        median: Median wall-clock time over the timed runs.
        iqr: Interquartile range (Q3 - Q1) of the timed runs.
        minimum: Fastest observed run.
        repeats: Number of timed runs.
    """

    median: float
    iqr: float
    minimum: float
    repeats: int


class Quality(NamedTuple):
    """Solution quality for a single result.

    Attributes:
        max_violation: Largest constraint violation (<= 0 means feasible).
        objective: Objective value attained by the solution.
    """

    max_violation: float
    objective: float


def _summarize(times: Sequence[float]) -> TimingStats:
    """Build a :class:`TimingStats` from raw wall-clock samples."""
    arr = np.asarray(times, dtype=float)
    q1, q3 = np.percentile(arr, [25.0, 75.0])
    return TimingStats(
        median=float(np.median(arr)),
        iqr=float(q3 - q1),
        minimum=float(arr.min()),
        repeats=len(arr),
    )


def time_repeat(
    func: Callable[[], Any],
    repeat: int = 5,
    warmup: int = 1,
) -> Tuple[TimingStats, List[Any]]:
    """Run ``func`` repeatedly and return robust timing statistics.

    The callable is invoked ``warmup`` times (untimed) to absorb any one-off
    setup or compilation cost, then ``repeat`` times with the wall-clock time
    of each run recorded.  The return values of the timed runs are collected
    so the caller can inspect the solution without a separate solve.

    Args:
        func: Zero-argument callable to benchmark.
        repeat: Number of timed runs (>= 1).
        warmup: Number of untimed warm-up runs (>= 0).

    Returns:
        A pair ``(stats, results)`` where ``stats`` holds the median, IQR and
        minimum wall-clock time and ``results`` is the list of return values
        from the timed runs.

    Raises:
        ValueError: If ``repeat`` is not positive.
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    for _ in range(warmup):
        func()

    times: List[float] = []
    results: List[Any] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        results.append(func())
        times.append(time.perf_counter() - t0)

    return _summarize(times), results


def time_alternating(
    funcs: Sequence[Callable[[], Any]],
    repeat: int = 5,
    warmup: int = 1,
) -> List[Tuple[TimingStats, List[Any]]]:
    """Time several callables, alternating between them on every round.

    Interleaving the runs (f0, f1, f0, f1, ...) removes any systematic bias
    from machine load drifting during a long benchmark.  Warm-up runs are
    likewise interleaved.

    Args:
        funcs: Callables to benchmark, in the order they should be reported.
        repeat: Number of timed rounds (each callable is timed once per round).
        warmup: Number of untimed warm-up rounds.

    Returns:
        One ``(stats, results)`` pair per callable, in the input order.  See
        :func:`time_repeat` for the meaning of each element.

    Raises:
        ValueError: If ``funcs`` is empty or ``repeat`` is not positive.
    """
    if not funcs:
        raise ValueError("funcs must be non-empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    for _ in range(warmup):
        for func in funcs:
            func()

    times: List[List[float]] = [[] for _ in funcs]
    results: List[List[Any]] = [[] for _ in funcs]
    for _ in range(repeat):
        for i, func in enumerate(funcs):
            t0 = time.perf_counter()
            results[i].append(func())
            times[i].append(time.perf_counter() - t0)

    return [(_summarize(times[i]), results[i]) for i in range(len(funcs))]


def quality_from_violations(
    violations: Sequence[float],
    objective: float,
) -> Quality:
    """Summarize solution quality from constraint residuals and objective.

    Args:
        violations: Constraint residuals ``gᵢ(x)`` where ``gᵢ(x) <= 0`` is
            satisfied.  Positive entries are constraint violations.
        objective: Objective value attained by the solution.

    Returns:
        A :class:`Quality` with the maximum violation (clipped at 0 so a
        feasible point reports ``0`` rather than a negative slack) and the
        objective value.
    """
    arr = np.asarray(violations, dtype=float)
    max_violation = float(np.maximum(arr, 0.0).max()) if arr.size else 0.0
    return Quality(max_violation=max_violation, objective=float(objective))


def max_violation(violations: Sequence[float]) -> float:
    """Return the largest constraint violation in ``violations`` (clipped at 0).

    Args:
        violations: Constraint residuals where ``<= 0`` is satisfied.

    Returns:
        ``max(max(violations), 0)`` as a float.
    """
    return quality_from_violations(violations, 0.0).max_violation
