"""
Type definitions, abstract base classes, and type aliases for the ellipsoid method.

This module provides:
    - ArrayType: A generic TypeVar bound to numpy.ndarray for type-safe array operations
    - Cut-related type aliases (SingleCut, ParallelCut, CutChoice, Cut, Num)
    - Abstract oracle interfaces defining the contract between cutting-plane
      algorithms and problem-specific feasibility/optimization logic
    - SearchSpace: Abstract interface for ellipsoidal search spaces
"""

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
    """Optimization oracle for convex optimization problems.

    Implement assess_optim to assess optimality at a candidate point and
    optionally return an improved objective value.
    """

    @abstractmethod
    def assess_optim(
        self, x_center: ArrayType, gamma: Any
    ) -> Tuple[Cut, Optional[float]]:
        """Assess optimality of candidate solution x_center at level gamma.

        Args:
            x_center: Current center point of the search space.
            gamma: Current best objective value.

        Returns:
            A tuple (cut, gamma_new) where cut is a separating hyperplane,
            and gamma_new is an improved objective value (or None if not improved).
        """
        ...


class OracleOptimQ(Generic[ArrayType]):
    """Optimization oracle for discrete/quantized convex optimization.

    Implement assess_optim_q to handle quantized solutions with a retry
    mechanism for discrete feasibility checks.
    """

    @abstractmethod
    def assess_optim_q(
        self, x_center: ArrayType, gamma: Any, retry: bool
    ) -> Tuple[Cut, ArrayType, Optional[float], bool]:
        """Assess optimality with quantized/rounded solutions.

        Args:
            x_center: Current continuous center point.
            gamma: Current best objective value.
            retry: Whether this is a retry with a rounded discrete solution.

        Returns:
            Tuple of (cut, x_q, gamma_new, more_alt) where:
                - cut: Separating hyperplane
                - x_q: Evaluation point (continuous or rounded)
                - gamma_new: Improved objective or None
                - more_alt: Whether alternative cuts remain available
        """
        ...


class OracleBS(ABC):
    """Binary search oracle for monotonic objectives.

    Implement assess_bs to test feasibility at a given parameter value,
    enabling binary search over the parameter space.
    """

    @abstractmethod
    def assess_bs(self, gamma: Any) -> bool:
        """Test feasibility at the given parameter value.

        Args:
            gamma: Parameter value to test.

        Returns:
            True if feasible, False otherwise.
        """
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
