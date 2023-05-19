from .ell_config import CutStatus

from abc import abstractmethod, ABC
from collections.abc import MutableSequence
from typing import Optional, Tuple, Union
from typing import TYPE_CHECKING

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
    def update(self, tea: Num) -> None:
        """update t

        Args:
            tea (Num): _description_
        """
        pass


class OracleOptim(ABC):
    @abstractmethod
    def assess_optim(
        self, xc: ArrayType, tea: float  # what?
    ) -> Tuple[Cut, Optional[float]]:
        """assessment of optimization

        Args:
            xc (ArrayType): _description_
            tea (float): _description_

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
        self, xc: ArrayType, tea: float, retry: bool
    ) -> Tuple[Cut, ArrayType, Optional[float], bool]:
        """assessment of optimization (discrete)

        Args:
            xc (ArrayType): _description_
            tea (float): _description_
            retry (bool): _description_

        Returns:
            Tuple[Cut, ArrayType, Optional[float], bool]: _description_
        """
        pass


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, tea: Num) -> bool:
        """assessment of the binary search

        Args:
            tea (Num): _description_

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

