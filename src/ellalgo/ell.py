"""
Ell Class

This code defines a class called Ell which represents an ellipsoidal search space. The purpose of this class is to provide methods for updating and manipulating an ellipsoid, which is a mathematical shape used in certain optimization algorithms.

The Ell class takes two main inputs when initialized: a value (which can be a number or a list of numbers) and an array xc. These inputs define the initial shape and position of the ellipsoid. The class doesn't produce a specific output on its own, but rather provides methods that can be used to modify and query the ellipsoid's state.

The class achieves its purpose by maintaining several internal attributes that represent the ellipsoid's properties, such as its center (_xc), a matrix (_mq), and scaling factors (_kappa and _tsq). It then provides methods to update these properties based on different types of "cuts" to the ellipsoid.

The main functionality of the Ell class revolves around three update methods: update_bias_cut, update_central_cut, and update_q. These methods take a "cut" as input, which is essentially a direction and a value that determine how to modify the ellipsoid. The cuts are used to shrink or reshape the ellipsoid, which is a key operation in certain optimization algorithms.

The core logic of these update methods is implemented in the private _update_core method. This method applies the cut to the ellipsoid by performing a series of mathematical operations. It calculates new values for the ellipsoid's center and shape based on the input cut and a specified cut strategy.

An important aspect of the code is its use of numpy, a library for numerical computations in Python. The class uses numpy arrays and matrix operations to efficiently perform the necessary calculations.

The class also includes some helper methods like xc() and tsq() that allow access to certain properties of the ellipsoid. These can be used to query the current state of the ellipsoid during an optimization process.

Overall, this code provides a flexible and efficient way to represent and manipulate an ellipsoidal search space, which is a crucial component in certain types of optimization algorithms. The class encapsulates the complex mathematics involved in these operations, providing a clean interface for users of the class to work with ellipsoids in their algorithms.
"""

from typing import Callable, Tuple, Union

import numpy as np

from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import ArrayType, SearchSpace2, SearchSpaceQ

Mat = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]


# The `Ell` class represents an ellipsoidal search space.
class Ell(SearchSpace2[ArrayType], SearchSpaceQ[ArrayType]):
    no_defer_trick: bool = False

    _mq: Mat
    _xc: ArrayType
    _kappa: float
    _tsq: float
    helper: EllCalc

    def __init__(self, val, xc: ArrayType) -> None:
        """
        The function initializes an object with given values and attributes.

        :param val: The parameter `val` can be either an integer, a float, or a list of numbers. If it
            is an integer or a float, it represents the value of kappa. If it is a list of numbers, it
            represents the diagonal elements of a matrix, mq

        :param xc: The parameter `xc` is of type `ArrayType`, which suggests that it is an array-like
            object. It is used to store the values of `xc` in the `__init__` method. The length of `xc` is
            calculated using `len(xc)` and stored in the variable

        :type xc: ArrayType
        """
        ndim = len(xc)
        self.helper = EllCalc(ndim)
        self._xc = xc
        self._tsq = 0.0
        if isinstance(val, (int, float)):
            self._kappa = val
            self._mq = np.eye(ndim)
        else:
            self._kappa = 1.0
            self._mq = np.diag(val)

    def xc(self) -> ArrayType:
        """
        The function `xc` returns the value of the `_xc` attribute.
        :return: The method `xc` is returning the value of the attribute `_xc`.
        """
        return self._xc

    def set_xc(self, xc: ArrayType) -> None:
        """
        The function sets the value of the variable `_xc` to the input `x`.

        :param x: The parameter `x` is of type `ArrayType`
        :type x: ArrayType
        """
        self._xc = xc

    def tsq(self) -> float:
        """
        The function `tsq` returns the measure of the distance between `xc` and `x*`.
        :return: The method is returning a float value, which represents the measure of the distance between xc and x*.
        """
        return self._tsq

    def update_bias_cut(self, cut) -> CutStatus:
        """
        The function `update_bias_cut` is an implementation of the `SearchSpace` interface that updates the
        ellipsoid based on a given deep-cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents some kind of cut
        :return: a `CutStatus` object.

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell.update_bias_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel)

    def update_central_cut(self, cut) -> CutStatus:
        """
        The function `update_central_cut` is an implementation of the `SearchSpace` interface that updates the
        ellipsoid based on a given central-cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents a cut
        :return: a `CutStatus` object.

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 0.0)
            >>> status = ell.update_central_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_central_cut)

    def update_q(self, cut) -> CutStatus:
        """
        The function `update_q` is an implementation of the `SearchSpaceQ` interface that updates the
        ellipsoid based on a given non-central cut (deep or shallow).

        :param cut: The `cut` parameter is of type `_type_` and it represents the cut that needs to be updated
        :return: a `CutStatus` object.

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), -0.01)
            >>> status = ell.update_q(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_q)

    # private:

    def _update_core(self, cut, cut_strategy: Callable) -> CutStatus:
        """
        The `_update_core` function updates an ellipsoid by applying a cut and a cut strategy.

        :param cut: The `cut` parameter is of type `_type_` and represents the cut to be applied to the
            ellipsoid. The specific type of `_type_` is not specified in the code snippet provided

        :param cut_strategy: The `cut_strategy` parameter is a callable object that represents the
            strategy for determining the cut status. It takes two arguments: `beta` and `tsq`. `beta` is a
            scalar value and `tsq` is a scalar value representing the squared norm of the current cut.

        :type cut_strategy: Callable

        :return: a `CutStatus` object.

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel)
            >>> print(status)
            CutStatus.Success

            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel_central_cut)
            >>> print(status)
            CutStatus.Success
        """
        grad, beta = cut
        grad_t = self._mq @ grad  # n^2 multiplications
        omega = grad.dot(grad_t)  # n multiplications
        self._tsq = self._kappa * omega

        status, result = cut_strategy(beta, self._tsq)

        if result is None:
            return status

        rho, sigma, delta = result

        self._xc -= (rho / omega) * grad_t
        self._mq -= (sigma / omega) * np.outer(grad_t, grad_t)
        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
