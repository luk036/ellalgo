"""
Configuration types for the ellipsoid method.

This module defines the core enumeration and options data classes used
throughout the ellipsoid method implementation.

- CutStatus: Outcome status of a cutting-plane update
- Options: Algorithm control parameters (iterations, tolerance, verbosity)
"""

from dataclasses import dataclass
from enum import Enum


class CutStatus(Enum):
    """Outcome status of an ellipsoid update operation.

    Returned by update methods to indicate whether the cut was applied
    successfully, or why it failed.

    Attributes:
        Success: Cut was applied successfully; ellipsoid was updated.
        NoSoln: Cut is infeasible (no solution exists in the remaining region).
        NoEffect: Cut had no effect on the ellipsoid (e.g., numerical degeneracy).
        Unknown: Fallback status for unclassified outcomes.
    """

    Success = 0
    NoSoln = 1
    NoEffect = 2
    Unknown = 3


class SolverStatus(Enum):
    """Termination status of a cutting-plane / binary-search solve.

    Attributes:
        Success: A solution (or converged bracket) was produced.
        Infeasible: The search space was exhausted; no solution exists.
        MaxIters: The iteration cap was reached before terminating.
        Stalled: The search space stopped shrinking at floating-point
            resolution before the tolerance was met (no further progress is
            representable). This is a *converged* stop, not a failure.
        Failed: A numerical breakdown was detected (e.g. the shape matrix lost
            positive definiteness, or an update returned ``CutStatus.Unknown``).
            Distinct from ``Infeasible`` so the silent-failure mode is loud.
    """

    Success = 0
    Infeasible = 1
    MaxIters = 2
    Stalled = 3
    Failed = 4


@dataclass
class Options:
    """Control parameters for cutting-plane algorithms.

    Attributes:
        max_iters: Maximum number of iterations before giving up.
        tolerance: Convergence threshold; algorithm stops when tsq < tolerance.
        verbose: If True, print progress information during execution.
        stall_guard: If True, the ellipsoid drivers stop with
            ``SolverStatus.Stalled`` once ``tsq`` fails to decrease beyond
            floating-point resolution. Pairing an absolute threshold with a
            stall test is required when the quantity scales with the problem
            data, otherwise the loop runs to ``max_iters`` for no gain.
        check_pd: If True, enable an ``O(n^3)`` positive-definiteness check of
            the shape matrix on every update (the direct ``Ell`` path can lose
            definiteness silently). Off by default for performance; intended
            for tests, debugging, and small problems. The cheap algebraic
            invariants (``sigma <= 1``, ``delta > 0``, ``kappa > 0``) are always
            enforced regardless of this flag.
    """

    max_iters: int = 2000
    tolerance: float = 1e-20
    verbose: bool = False
    stall_guard: bool = True
    check_pd: bool = False
