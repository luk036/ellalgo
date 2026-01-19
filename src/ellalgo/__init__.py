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
