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
    """CutStatus

    The `CutStatus` enum defines a set of constant values that represent different statuses that can result from a cut operation. A cut is likely some optimization operation that partitions or divides a problem into smaller pieces.

    This enum has four possible values:

    Success - Indicates the cut operation succeeded
    NoSoln - Indicates the cut did not yield a valid solution
    NoEffect - The cut had no effect on improving the optimization
    Unknown - The status is unknown or unclear

    Examples:
        >>> from ellalgo.ell_config import CutStatus
        >>> CutStatus.Success
        <CutStatus.Success: 0>
        >>> CutStatus.NoSoln.value
        1
    """

    Success = 0
    NoSoln = 1
    NoEffect = 2
    Unknown = 3


# The class "Options" defines two attributes, "max_iters" and "tolerance", with default values of 2000 and
# 1e-8 respectively.
class Options:
    """Options

    The class "Options" defines two attributes, "max_iters" and "tolerance", with default values of 2000 and
    1e-8 respectively.

    Examples:
        >>> from ellalgo.ell_config import Options
        >>> options = Options()
        >>> options.max_iters
        2000
        >>> options.tolerance
        1e-20
    """

    max_iters: int = 2000  # maximum number of iterations
    tolerance: float = 1e-20  # error tolerance
