from typing import Callable, Tuple, Union

import numpy as np

from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import ArrayType, SearchSpace, SearchSpaceQ

Matrix = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]


# The `EllStable` class represents an ellipsoidal search space with stability properties.
class EllStable(SearchSpace[ArrayType], SearchSpaceQ[ArrayType]):
    no_defer_trick: bool = False

    _mq: Matrix
    _xc: ArrayType
    _kappa: float
    _tsq: float
    _ndim: int
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
        self._ndim = ndim
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

    def set_xc(self, x: ArrayType) -> None:
        """
        The function sets the value of the variable `_xc` to the input `x`.

        :param x: The parameter `x` is of type `ArrayType`
        :type x: ArrayType
        """
        self._xc = x

    def tsq(self) -> float:
        """
        The function `tsq` returns the measure of the distance between `xc` and `x*`.
        :return: The method is returning a float value, which represents the measure of the distance between xc and x*.
        """
        return self._tsq

    def update_bias_cut(self, cut) -> CutStatus:
        """
        The function `update_bias_cut` is an implementation of the `SearchSpace` interface that updates the
        cut status based on a given cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents some kind of cut
        :return: a `CutStatus` object.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell.update_bias_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel)

    def update_central_cut(self, cut) -> CutStatus:
        """
        The function `update_central_cut` is an implementation of the `SearchSpace` interface that updates the
        cut status based on a given cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents a cut
        :return: a `CutStatus` object.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 0.0)
            >>> status = ell.update_central_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_central_cut)

    def update_q(self, cut) -> CutStatus:
        """
        The function `update_q` is an implementation of the `SearchSpaceQ` interface that updates the
        cut status based on a given cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents the cut that needs to be
            updated
        :return: a `CutStatus` object.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
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

        Reference:
            Gill, Murray, and Wright, "Practical Optimization", p43.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel)
            >>> print(status)
            CutStatus.Success

            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 0.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel_central_cut)
            >>> print(status)
            CutStatus.Success
        """
        g, beta = cut

        # calculate inv(L)*g: (n-1)*n/2 multiplications
        inv_lower_g = g.copy()  # initially

        for j in range(self._ndim - 1):
            for i in range(j + 1, self._ndim):
                self._mq[j, i] = self._mq[i, j] * inv_lower_g[j]
                # keep for rank-one update
                inv_lower_g[i] -= self._mq[j, i]

        # calculate inv(D)*inv(L)*g: n
        inv_diag_inv_lower_g = inv_lower_g.copy()  # initially
        for i in range(self._ndim):
            inv_diag_inv_lower_g[i] *= self._mq[i, i]

        # print(inv_diag_inv_lower_g)
        # calculate omega: n
        gg_t = inv_lower_g * inv_diag_inv_lower_g
        omega = sum(gg_t)

        self._tsq = self._kappa * omega  # need for helper

        status, result = cut_strategy(beta, self._tsq)

        if result is None:
            return status

        rho, sigma, delta = result

        # calculate Q*g = inv(L')*inv(D)*inv(L)*g : (n-1)*n/2
        g_t = inv_diag_inv_lower_g.copy()  # initially
        for i in range(self._ndim - 1, 0, -1):
            for j in range(i, self._ndim):
                g_t[i - 1] -= self._mq[j, i - 1] * g_t[j]  # TODO

        # print(g_t)
        # calculate xc: n
        self._xc -= (rho / omega) * g_t

        # rank-one update: 3*n + (n-1)*n/2
        # r = self._sigma / omega
        mu = sigma / (1.0 - sigma)
        oldt = omega / mu  # initially
        v = g.copy()
        for j in range(self._ndim):
            p = v[j]
            temp = inv_diag_inv_lower_g[j]
            newt = oldt + p * temp
            beta2 = temp / newt
            self._mq[j, j] *= oldt / newt  # update invD
            for k in range(j + 1, self._ndim):
                v[k] -= self._mq[j, k]
                self._mq[k, j] += beta2 * v[k]
            oldt = newt

        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
