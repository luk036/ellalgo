from enum import Enum


# The CutStatus enum defines a set of constant values that represent different statuses that can result from a cut operation. A cut is likely some optimization operation that partitions or divides a problem into smaller pieces.
#
# This enum has four possible values:
#
# Success - Indicates the cut operation succeeded
# NoSoln - Indicates the cut did not yield a valid solution
# NoEffect - The cut had no effect on improving the optimization
# Unknown - The status is unknown or unclear
class CutStatus(Enum):
    """
    Represents the outcome of a single cutting-plane iteration.

    The `CutStatus` enum provides a set of symbolic names to represent the status
    of a cut operation in an optimization algorithm. Each status indicates how the
    search space was affected by the cut.

    - `Success`: The cutting-plane was successfully generated and applied,
      reducing the search space.
    - `NoSoln`: The problem is determined to be infeasible. This means there is
      no solution that satisfies all the constraints.
    - `NoEffect`: The cutting-plane had no effect on the search space. This may
      happen if the cut is redundant or lies outside the current search region.
    - `Unknown`: The status of the cut operation is unknown or could not be
      determined.

    Examples:
        >>> from ellalgo.ell_config import CutStatus
        >>> CutStatus.Success
        <CutStatus.Success: 0>
        >>> CutStatus.NoSoln
        <CutStatus.NoSoln: 1>
        >>> CutStatus.NoEffect
        <CutStatus.NoEffect: 2>
        >>> CutStatus.Unknown
        <CutStatus.Unknown: 3>
    """

    Success = 0
    NoSoln = 1
    NoEffect = 2
    Unknown = 3


# The class "Options" defines two attributes, "max_iters" and "tolerance", with default values of 2000 and
# 1e-8 respectively.
class Options:
    """
    Configuration options for the optimization algorithm.

    The `Options` class holds parameters that control the behavior of the cutting-plane
    methods. These options allow for fine-tuning the algorithm's termination
    criteria and precision.

    Attributes:
        max_iters (int): The maximum number of iterations to perform before stopping.
            This prevents the algorithm from running indefinitely.
        tolerance (float): The numerical tolerance for convergence. The algorithm
            stops when the size of the search space (e.g., `tsq`) is smaller than
            this value, indicating that a solution has been found to the desired
            precision.

    Examples:
        >>> from ellalgo.ell_config import Options
        >>> options = Options()
        >>> options.max_iters
        2000
        >>> options.tolerance
        1e-20
        >>> options.max_iters = 1000
        >>> options.max_iters
        1000
    """

    max_iters: int = 2000  # maximum number of iterations
    tolerance: float = 1e-20  # error tolerance
