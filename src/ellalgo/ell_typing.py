from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, MutableSequence, Optional, Tuple, Union

from .ell_config import CutStatus

if TYPE_CHECKING:
    import numpy as np

    ArrayType = np.ndarray
else:
    from typing import Any

    ArrayType = Any

CutChoice = Union[float, MutableSequence]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]
Num = Union[float, int]


class OracleFeas(ABC):
    @abstractmethod
    def assess_feas(self, xc: ArrayType) -> Optional[Cut]:
        """assessment of feasibility

        Args:
            xc (ArrayType): _description_

        Returns:
            Optional[Cut]: _description_
        """
        pass


class OracleFeas2(OracleFeas):
    @abstractmethod
    def update(self, target) -> None:
        """update t

        Args:
            target (Any): _description_
        """
        pass


class OracleOptim(ABC):
    @abstractmethod
    def assess_optim(self, xc: ArrayType, target) -> Tuple[Cut, Optional[float]]:
        """assessment of optimization

        Args:
            xc (ArrayType): _description_
            target (Any): _description_

        Returns:
            Tuple[Cut, Optional[float]]: _description_
        """
        pass


class OracleFeasQ(ABC):
    @abstractmethod
    def assess_feas_q(
        self, xc: ArrayType, retry: bool
    ) -> Tuple[Optional[Cut], Optional[ArrayType], bool]:
        """assessment of feasibility (discrete)

        Args:
            xc (ArrayType): _description_
            retry (bool): _description_

        Returns:
            Tuple[Optional[Cut], Optional[ArrayType], bool]: _description_
        """
        pass


class OracleOptimQ(ABC):
    @abstractmethod
    def assess_optim_q(
        self, xc: ArrayType, target, retry: bool
    ) -> Tuple[Cut, ArrayType, Optional[float], bool]:
        """assessment of optimization (discrete)

        Args:
            xc (ArrayType): _description_
            target (Any): _description_
            retry (bool): _description_

        Returns:
            Tuple[Cut, ArrayType, Optional[float], bool]: _description_
        """
        pass


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, target) -> bool:
        """assessment of the binary search

        Args:
            target (Any): _description_

        Returns:
            bool: _description_
        """
        pass


class SearchSpace(ABC):
    @abstractmethod
    def update_dc(self, cut: Cut) -> CutStatus:
        """update of deep-cut

        Args:
            cut (Cut): _description_

        Returns:
            CutStatus: _description_
        """
        pass

    @abstractmethod
    def update_cc(self, cut: Cut) -> CutStatus:
        """update of central cut

        Args:
            cut (Cut): _description_

        Returns:
            CutStatus: _description_
        """
        pass

    @abstractmethod
    def xc(self) -> ArrayType:
        pass

    @abstractmethod
    def tsq(self) -> float:
        pass


class SearchSpaceQ(ABC):
    @abstractmethod
    def update_q(self, cut: Cut) -> CutStatus:
        """update of shadow cut (discrete)

        Args:
            cut (Cut): _description_

        Returns:
            CutStatus: _description_
        """
        pass

    @abstractmethod
    def xc(self) -> ArrayType:
        pass

    @abstractmethod
    def tsq(self) -> float:
        pass


class SearchSpace2(SearchSpace):
    @abstractmethod
    def set_xc(self, xc: ArrayType) -> None:
        pass
