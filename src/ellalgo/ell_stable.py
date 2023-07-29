import numpy as np
from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import SearchSpace, SearchSpaceQ
from typing import Tuple, Union, Callable

Matrix = np.ndarray
ArrayType = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]


# The `EllStable` class represents an ellipsoidal search space with stability properties.
class EllStable(SearchSpace, SearchSpaceQ):
    no_defer_trick: bool = False

    _mq: Matrix
    _xc: ArrayType
    _kappa: float
    _tsq: float
    _ndim: int
    _helper: EllCalc

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
        self._helper = EllCalc(ndim)
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
        """_summary_

        Args:
            x (ArrayType): _description_
        """
        self._xc = x

    def tsq(self) -> float:
        """Measure of the distance between xc and x*

        Returns:
            float: [description]
        """
        return self._tsq

    def update_dc(self, cut) -> CutStatus:
        """Implement SearchSpace interface

        Args:
            cut (_type_): _description_

        Returns:
            CutStatus: _description_
        """
        return self._update_core(cut, self._helper.calc_single_or_ll)

    def update_cc(self, cut) -> CutStatus:
        """Implement SearchSpace interface

        Args:
            cut (_type_): _description_

        Returns:
            CutStatus: _description_
        """
        return self._update_core(cut, self._helper.calc_single_or_ll_cc)

    def update_q(self, cut) -> CutStatus:
        """Implement SearchSpaceQ interface

        Args:
            cut (_type_): _description_

        Returns:
            CutStatus: _description_
        """
        return self._update_core(cut, self._helper.calc_single_or_ll_q)

    # private:

    def _update_core(self, cut, cut_strategy: Callable) -> CutStatus:
        """Update ellipsoid by cut

        Args:
            cut (_type_): _description_
            cut_strategy (Callable): _description_

        Returns:
            CutStatus: _description_

        Reference:
            Gill, Murray, and Wright, "Practical Optimization", p43.
        """
        g, beta = cut

        # calculate inv(L)*g: (n-1)*n/2 multiplications
        invLg = g.copy()  # initially

        for j in range(self._ndim - 1):
            for i in range(j + 1, self._ndim):
                self._mq[j, i] = self._mq[i, j] * invLg[j]
                # keep for rank-one update
                invLg[i] -= self._mq[j, i]

        # calculate inv(D)*inv(L)*g: n
        invDinvLg = invLg.copy()  # initially
        for i in range(self._ndim):
            invDinvLg[i] *= self._mq[i, i]

        # print(invDinvLg)
        # calculate omega: n
        gg_t = invLg * invDinvLg
        omega = sum(gg_t)

        self._tsq = self._kappa * omega  # need for helper

        status, rho, sigma, delta = cut_strategy(beta, self._tsq)

        # if central_cut:
        #     status = self._helper.calc_single_or_ll_cc(beta)
        # else:
        #     status = self._helper.calc_single_or_ll(beta)

        if status != CutStatus.Success:
            return status

        # calculate Q*g = inv(L')*inv(D)*inv(L)*g : (n-1)*n/2
        g_t = invDinvLg.copy()  # initially
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
            # temp = p * self._mq[j, j]
            temp = invDinvLg[j]
            newt = oldt + p * temp
            beta2 = temp / newt
            self._mq[j, j] *= oldt / newt  # update invD
            for k in range(j + 1, self._ndim):
                # v[k] -= p * self._mq[k, j]
                v[k] -= self._mq[j, k]
                self._mq[k, j] += beta2 * v[k]
            oldt = newt

        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
