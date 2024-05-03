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


class OracleFeas2(OracleFeas[ArrayType]):
    @abstractmethod
    def update(self, gamma) -> None:
        """
        The `update` function updates a gamma object.

        :param gamma: The `gamma` parameter is of type `Any`, which means it can accept any type of value.
            It is used as an argument to update the gamma object
        """


class OracleOptim(Generic[ArrayType]):
    @abstractmethod
    def assess_optim(self, xc: ArrayType, gamma) -> Tuple[Cut, Optional[float]]:
        """
        The `assess_optim` function assesses the feasibility based on the given `xc` and `gamma`
        parameters.

        :param xc: An array of values that represents the current solution or point in the optimization
            process

        :type xc: ArrayType

        :param gamma: The `gamma` parameter is the value that we are trying to optimize or minimize. It
            could be a numerical value, a function, or any other type of object that represents the optimization
            goal
        """


# class OracleFeasQ(Generic[ArrayType]):
#     @abstractmethod
#     def assess_feas_q(
#         self, xc: ArrayType, retry: bool
#     ) -> Tuple[Optional[Cut], Optional[ArrayType], bool]:
#         """assessment of feasibility (discrete)
#
#         The function assess_feas_q assesses the feasibility of a given input and returns a tuple containing
#         a cut, an array, and a boolean value.
#
#         :param xc: An array of some type. It represents a variable or a set of variables that need to be
#             assessed for feasibility
#
#         :type xc: ArrayType
#
#         :param retry: A boolean flag indicating whether to retry the assessment if it fails initially
#
#         :type retry: bool
#         """


class OracleOptimQ(Generic[ArrayType]):
    @abstractmethod
    def assess_optim_q(
        self, xc: ArrayType, gamma, retry: bool
    ) -> Tuple[Cut, ArrayType, Optional[float], bool]:
        """assessment of optimization (discrete)

        The function `assess_optim_q` assesses the feasibility of a design variable and returns a tuple
        containing a cut, an array, an optional float, and a boolean value.

        :param xc: An array or list representing the current solution or configuration being assessed for
            optimization

        :type xc: ArrayType

        :param gamma: The `gamma` parameter is the desired value or condition that the optimization
            algorithm is trying to achieve. It could be a specific value, a range of values, or a certain
            condition that needs to be satisfied

        :param retry: A boolean flag indicating whether to retry the optimization if it fails

        :type retry: bool
        """


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, gamma) -> bool:
        """
        The `assess_bs` function is a binary search assessment function that takes a gamma value as input
        and returns a boolean value.

        :param gamma: The gamma parameter is the value that we are searching for in the binary search
        """


# The `SearchSpace` class is an abstract base class that defines methods for updating deep-cut and
# central cut, as well as accessing the xc and tsq attributes.
class SearchSpace(Generic[ArrayType]):
    @abstractmethod
    def update_bias_cut(self, cut: Cut) -> CutStatus:
        """
        The `update_bias_cut` function is an abstract method that takes a `Cut` object as input and returns
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
        """
        The function `xc` returns the value of the `_xc` attribute.
        :return: The method `xc` is returning the value of the attribute `_xc`.
        """

    @abstractmethod
    def tsq(self) -> float:
        """
        The function `tsq` returns the measure of the distance between `xc` and `x*`.
        :return: The method is returning a float value, which represents the measure of the distance between xc and x*.
        """


class SearchSpaceQ(Generic[ArrayType]):
    @abstractmethod
    def update_q(self, cut: Cut) -> CutStatus:
        """
        The `update_q` function is an abstract method that updates a shadow cut and returns a `CutStatus`
        object.

        :param cut: The `cut` parameter is an object of type `Cut`
        :type cut: Cut
        """

    @abstractmethod
    def xc(self) -> ArrayType:
        """
        The function `xc` returns the value of the `_xc` attribute.
        :return: The method `xc` is returning the value of the attribute `_xc`.
        """

    @abstractmethod
    def tsq(self) -> float:
        """
        The function `tsq` returns the measure of the distance between `xc` and `x*`.
        :return: The method is returning a float value, which represents the measure of the distance between xc and x*.
        """


class SearchSpace2(SearchSpace[ArrayType]):
    @abstractmethod
    def set_xc(self, xc: ArrayType) -> None:
        """
        The function sets the value of the variable `_xc` to the input `x`.

        :param x: The parameter `x` is of type `ArrayType`
        :type x: ArrayType
        """
