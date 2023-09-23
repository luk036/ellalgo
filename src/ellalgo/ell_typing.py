from abc import ABC, abstractmethod
from typing import Generic, MutableSequence, Optional, Tuple, TypeVar, Union

import numpy as np

from .ell_config import CutStatus

ArrayType = TypeVar("ArrayType", bound=np.ndarray)
CutChoice = Union[float, MutableSequence]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]
Num = Union[float, int]


class OracleFeas(Generic[ArrayType]):
    @abstractmethod
    def assess_feas(self, xc: ArrayType) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input and returns a cut if it is
        not feasible.

        :param xc: An array of type ArrayType
        :type xc: ArrayType
        """
        pass


class OracleFeas2(OracleFeas[ArrayType]):
    @abstractmethod
    def update(self, target) -> None:
        """
        The `update` function updates a target object.

        :param target: The `target` parameter is of type `Any`, which means it can accept any type of value.
        It is used as an argument to update the target object
        """
        pass


class OracleOptim(Generic[ArrayType]):
    @abstractmethod
    def assess_optim(self, xc: ArrayType, target) -> Tuple[Cut, Optional[float]]:
        """
        The `assess_optim` function assesses the feasibility based on the given `xc` and `target`
        parameters.

        :param xc: An array of values that represents the current solution or point in the optimization
        process
        :type xc: ArrayType
        :param target: The `target` parameter is the value that we are trying to optimize or minimize. It
        could be a numerical value, a function, or any other type of object that represents the optimization
        goal
        """
        pass


class OracleFeasQ(Generic[ArrayType]):
    @abstractmethod
    def assess_feas_q(
        self, xc: ArrayType, retry: bool
    ) -> Tuple[Optional[Cut], Optional[ArrayType], bool]:
        """assessment of feasibility (discrete)

        The function assess_feas_q assesses the feasibility of a given input and returns a tuple containing
        a cut, an array, and a boolean value.

        :param xc: An array of some type. It represents a variable or a set of variables that need to be
        assessed for feasibility
        :type xc: ArrayType
        :param retry: A boolean flag indicating whether to retry the assessment if it fails initially
        :type retry: bool
        """
        pass


class OracleOptimQ(Generic[ArrayType]):
    @abstractmethod
    def assess_optim_q(
        self, xc: ArrayType, target, retry: bool
    ) -> Tuple[Cut, ArrayType, Optional[float], bool]:
        """assessment of optimization (discrete)

        The function `assess_optim_q` assesses the feasibility of a design variable and returns a tuple
        containing a cut, an array, an optional float, and a boolean value.

        :param xc: An array or list representing the current solution or configuration being assessed for
        optimization
        :type xc: ArrayType
        :param target: The `target` parameter is the desired value or condition that the optimization
        algorithm is trying to achieve. It could be a specific value, a range of values, or a certain
        condition that needs to be satisfied
        :param retry: A boolean flag indicating whether to retry the optimization if it fails
        :type retry: bool
        """
        pass


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, target) -> bool:
        """
        The `assess_bs` function is a binary search assessment function that takes a target value as input
        and returns a boolean value.

        :param target: The target parameter is the value that we are searching for in the binary search
        """
        pass


# The `SearchSpace` class is an abstract base class that defines methods for updating deep-cut and
# central cut, as well as accessing the xc and tsq attributes.
class SearchSpace(Generic[ArrayType]):
    @abstractmethod
    def update_deep_cut(self, cut: Cut) -> CutStatus:
        """
        The `update_deep_cut` function is an abstract method that takes a `Cut` object as input and returns
        a `CutStatus` object.
        
        :param cut: The `cut` parameter is an instance of the `Cut` class. It represents a deep-cut that
        needs to be updated
        :type cut: Cut
        """

    @abstractmethod
    def update_central_cut(self, cut: Cut) -> CutStatus:
        """
        The `update_central_cut` function is an abstract method that updates the central cut and returns the
        status of the cut.
        
        :param cut: The "cut" parameter is an instance of the Cut class. It represents the central cut that
        needs to be updated
        :type cut: Cut
        """

    @abstractmethod
    def xc(self) -> ArrayType:
        pass

    @abstractmethod
    def tsq(self) -> float:
        pass


class SearchSpaceQ(Generic[ArrayType]):
    @abstractmethod
    def update_q(self, cut: Cut) -> CutStatus:
        """
        The `update_q` function is an abstract method that updates a shadow cut and returns a `CutStatus`
        object.
        
        :param cut: The `cut` parameter is an object of type `Cut`
        :type cut: Cut
        """
        pass

    @abstractmethod
    def xc(self) -> ArrayType:
        pass

    @abstractmethod
    def tsq(self) -> float:
        pass


class SearchSpace2(SearchSpace[ArrayType]):
    @abstractmethod
    def set_xc(self, xc: ArrayType) -> None:
        pass
