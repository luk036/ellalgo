"""
Ellipsoid Method in Python

This package provides an implementation of the ellipsoid method for convex
optimization problems. The ellipsoid method is a polynomial-time algorithm
first introduced by L. G. Khachiyan in 1979.

Key Components:
    - Ell, EllStable: Ellipsoid search space representations
    - cutting_plane_feas, cutting_plane_optim: Cutting-plane algorithms
    - Oracle classes: Problem-specific feasibility and optimization oracles
    - LDLTMgr: LDLT factorization for Linear Matrix Inequality (LMI) constraints

The package supports:
    - Deep cuts, central cuts, and parallel cuts
    - Semidefinite programming via LMI oracles
    - Discrete optimization via quantized search spaces
    - Binary search for monotonic objectives

Examples:
    >>> import numpy as np
    >>> from ellalgo import Ell, cutting_plane_feas
    >>> # Solve a simple feasibility problem
    >>> class MyOracle:
    ...     def assess_feas(self, xc):
    ...         return (np.array([1.0, 1.0]), 0.0) if xc[0] + xc[1] > 0 else None
    >>> space = Ell(10.0, np.array([0.0, 0.0]))
    >>> x, niter = cutting_plane_feas(MyOracle(), space)
    >>> x is not None
    True

References:
    - Khachiyan, L. G. (1979). "A polynomial algorithm in linear programming"
    - Gill, Murray, and Wright, "Practical Optimization"
"""

import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >=3.9`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .conjugate_gradient import conjugate_gradient
from .cutting_plane import (
    bsearch,
    cutting_plane_feas,
    cutting_plane_optim,
    cutting_plane_optim_q,
)

# Public API exports
from .ell import Ell
from .ell_config import CutStatus, Options
from .ell_stable import EllStable
from .ell_typing import (
    ArrayType,
    Cut,
    CutChoice,
    Num,
    OracleBS,
    OracleFeas,
    OracleFeas2,
    OracleOptim,
    OracleOptimQ,
    SearchSpace,
    SearchSpace2,
    SearchSpaceQ,
)
from .oracles.lmi0_oracle import LMI0Oracle
from .oracles.lmi_oracle import LMIOracle
from .oracles.lowpass_oracle import LowpassOracle
from .oracles.profit_oracle import ProfitOracle, ProfitQOracle, ProfitRbOracle

__all__ = [
    "Ell",
    "EllStable",
    "cutting_plane_feas",
    "cutting_plane_optim",
    "cutting_plane_optim_q",
    "bsearch",
    "conjugate_gradient",
    "CutStatus",
    "Options",
    "ArrayType",
    "Cut",
    "CutChoice",
    "Num",
    "OracleBS",
    "OracleFeas",
    "OracleFeas2",
    "OracleOptim",
    "OracleOptimQ",
    "SearchSpace",
    "SearchSpace2",
    "SearchSpaceQ",
    "LMIOracle",
    "LMI0Oracle",
    "ProfitOracle",
    "ProfitRbOracle",
    "ProfitQOracle",
    "LowpassOracle",
]
