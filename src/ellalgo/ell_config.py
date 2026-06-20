"""
Configuration types for the ellipsoid method.

This module defines the core enumeration and options data classes used
throughout the ellipsoid method implementation.

- CutStatus: Outcome status of a cutting-plane update
- Options: Algorithm control parameters (iterations, tolerance, verbosity)
"""

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


class Options:
    """Control parameters for cutting-plane algorithms.

    Attributes:
        max_iters: Maximum number of iterations before giving up.
        tolerance: Convergence threshold; algorithm stops when tsq < tolerance.
        verbose: If True, print progress information during execution.
    """

    max_iters: int = 2000
    tolerance: float = 1e-20
    verbose: bool = False
