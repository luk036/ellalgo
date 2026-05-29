from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

from .ell_config import CutStatus

ArrayType = TypeVar("ArrayType", bound=np.ndarray)

# --- Cut types ---
# Single cut:  gᵀ(x - xc) + β ≤ 0
SingleCut = float
# Parallel cut:  β₀ ≤ gᵀ(x - xc) ≤ β₁
ParallelCut = Union[Tuple[float, Optional[float]], List[float]]
CutChoice = Union[float, ParallelCut]  # single or parallel cut
Cut = Tuple[ArrayType, CutChoice]
Num = Union[float, int]


class OracleFeas(Generic[ArrayType]):
    """Feasibility oracle with optional gamma update.

    Implement assess_feas to check feasibility.
    Override update() if the oracle needs to respond to gamma changes
    (used by BSearchAdaptor).
    """

    @abstractmethod
    def __init__(
        self, mat_f: List[np.ndarray], mat_b: Optional[np.ndarray] = None
    ) -> None:
        ...

    @abstractmethod
    def assess_feas(self, x_center: ArrayType) -> Optional[Cut]:
        ...

    def update(self, gamma: Any) -> None:
        """Default no-op. Override for gamma-sensitive oracles."""
        return


class OracleOptim(Generic[ArrayType]):
    @abstractmethod
    def assess_optim(
        self, x_center: ArrayType, gamma: Any
    ) -> Tuple[Cut, Optional[float]]:
        ...


class OracleOptimQ(Generic[ArrayType]):
    @abstractmethod
    def assess_optim_q(
        self, x_center: ArrayType, gamma: Any, retry: bool
    ) -> Tuple[Cut, ArrayType, Optional[float], bool]:
        ...


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, gamma: Any) -> bool:
        ...


class SearchSpace(Generic[ArrayType]):
    """Search space for cutting-plane methods.

    Unified interface that replaces SearchSpace, SearchSpaceQ, and SearchSpace2.
    - update_q() defaults to update_bias_cut; override for custom quantized behavior.
    - set_xc() defaults to no-op; override to allow external center updates.
    """

    @abstractmethod
    def __init__(self, val: Union[float, ArrayType], x_center: ArrayType) -> None:
        ...

    @abstractmethod
    def update_bias_cut(self, cut: Cut) -> CutStatus:
        ...

    @abstractmethod
    def update_central_cut(self, cut: Cut) -> CutStatus:
        ...

    def update_q(self, cut: Cut) -> CutStatus:
        """Quantized update. Default delegates to update_bias_cut."""
        return self.update_bias_cut(cut)

    @abstractmethod
    def xc(self) -> ArrayType:
        ...

    @abstractmethod
    def tsq(self) -> float:
        ...

    def set_xc(self, x_center: ArrayType) -> None:
        """Default no-op. Override to allow external center updates."""
        return
