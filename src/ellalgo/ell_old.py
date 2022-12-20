# -*- coding: utf-8 -*-
import math
from typing import Tuple, Union

import numpy as np

from .cutting_plane import CutStatus

Arr = Union[np.ndarray]


class Ell:
    """Ellipsoid Search Space

            Ell = {x | (x − xc)' Q^−1 (x − xc) ≤ κ}

    Returns:
        [type] -- [description]
    """

    # __slots__ = ('_n', '_c1', '_kappa', '_rho', '_sigma', '_delta', '_tsq',
    #              '_xc', '_Q', 'use_parallel_cut', 'no_defer_trick')

    def __init__(self, val: Union[Arr, float], x: Arr):
        """Construct a new Ell object

        Arguments:
            val (Union[Arr, float]): [description]
            x (Arr): [description]
        """
        self.use_parallel_cut = True
        self.no_defer_trick = False

        self._n = n = len(x)
        self._nSq = float(n * n)
        self._nPlus1 = float(n + 1)
        self._nMinus1 = float(n - 1)
        self._halfN = float(n) / 2.0
        self._halfNplus1 = self._nPlus1 / 2.0
        self._halfNminus1 = self._nMinus1 / 2.0
        self._c1 = self._nSq / (self._nSq - 1)
        self._c2 = 2.0 / self._nPlus1
        self._c3 = float(n) / self._nPlus1

        self._xc = x
        self._kappa = 1.0
        if np.isscalar(val):
            self._Q = np.eye(n)
            if self.no_defer_trick:
                self._Q *= val
            else:
                self._kappa = val
        else:
            self._Q = np.diag(val)

    def copy(self):
        """[summary]

        Returns:
            Ell: [description]
        """
        E = Ell(self._kappa, self.xc)
        E._Q = self._Q.copy()
        # E._c1 = self._c1
        E.use_parallel_cut = self.use_parallel_cut
        E.no_defer_trick = self.no_defer_trick
        return E

    @property
    def xc(self):
        """copy the whole array anyway

        Returns:
            [type]: [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x: Arr):
        """Set the xc object

        Arguments:
            x ([type]): [description]
        """
        self._xc = x

    # @property
    # def use_parallel_cut(self) -> bool:
    #     """[summary]

    #     Returns:
    #         bool: [description]
    #     """
    #     return self._use_parallel_cut

    # @use_parallel_cut.setter
    # def use_parallel_cut(self, b: bool):
    #     """[summary]

    #     Arguments:
    #         b (bool): [description]
    #     """
    #     self._use_parallel_cut = b

    def update(self, cut) -> Tuple[int, float]:
        """Update ellipsoid core function using the cut(s)

        Arguments:
            cut: cutting-plane

        Returns:
            Tuple[int, float]: [description]
        """
        return self.update_core(self.calc_ll, cut)

    def update_core(self, calc_ell, cut):
        """Update ellipsoid core function using the cut(s)

                grad' (x − xc) + beta ≤ 0

            Note: At most one square-root per iteration.

        Arguments:
            calc_ell ([type]): [description]
            cut (float): [description]

        Returns:
            status: 0: success
            tau: "volumn" of ellipsoid
        """
        grad, beta = cut
        grad_t = self._Q @ grad  # n^2 multiplications
        omega = grad @ grad_t  # n multiplications
        self._tsq = self._kappa * omega
        status = calc_ell(beta)
        if status != CutStatus.Success:
            return status, self._tsq
        self._xc -= (self._rho / omega) * grad_t  # n
        self._Q -= (self._sigma / omega) * np.outer(grad_t, grad_t)  # n*(n-1)/2
        if self.no_defer_trick:
            self._Q *= self._delta
        else:
            self._kappa *= self._delta
        return status, self._tsq

    def calc_ll(self, beta) -> CutStatus:
        """parallel or deep cut

        Arguments:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if np.isscalar(beta):
            return self.calc_dc(beta)
        if len(beta) < 2:  # unlikely
            return self.calc_dc(beta[0])
        return self.calc_ll_core(beta[0], beta[1])

    def calc_ll_core(self, b0: float, b1: float) -> CutStatus:
        """Calculate new ellipsoid under Parallel Cut

                g' (x − xc) + β0 ≤ 0
                g' (x − xc) + β1 ≥ 0

        Arguments:
            b0 (float): [description]
            b1 (float): [description]

        Returns:
            int: [description]
        """
        if self._tsq < b1 * b1 or not self.use_parallel_cut:
            return self.calc_dc(b0)
        if b1 < b0:
            return CutStatus.NoSoln  # no sol'n
        if b0 == 0:
            self.calc_ll_cc(b1)
            return CutStatus.Success

        b0b1 = b0 * b1
        if self._n * b0b1 < -self._tsq:  # unlikely
            return CutStatus.NoEffect  # no effect

        # parallel cut
        t1n = self._tsq - b1 * b1
        t0n = self._tsq - b0 * b0
        # t1 = self._tsq - b1sq
        bsum = b0 + b1
        bav = bsum / 2.0
        temp = self._n * bav * (b1 - b0)
        xi = math.sqrt(t0n * t1n + temp * temp)
        self._sigma = self._c3 + (self._tsq - b0b1 - xi) / (bsum * bav * self._nPlus1)
        self._rho = self._sigma * bav
        self._delta = self._c1 * ((t0n + t1n) / 2 + xi / self._n) / self._tsq
        return CutStatus.Success

    def calc_ll_cc(self, b1: float):
        """Calculate new ellipsoid under Parallel Cut, one of them is central

                g' (x − xc) ≤ 0
                g' (x − xc) + β1 ≥ 0

        Arguments:
            b1 (float): [description]
            b1sq (float): [description]
        """
        b1sqn = b1 * (b1 / self._tsq)
        xi = math.sqrt(1 - b1sqn + (self._halfN * b1sqn) ** 2)
        self._sigma = self._c3 + self._c2 * (1.0 - xi) / b1sqn
        self._rho = self._sigma * b1 / 2.0
        self._delta = self._c1 * (1.0 - b1sqn / 2.0 + xi / self._n)

    def calc_dc(self, beta: float) -> CutStatus:
        """Calculate new ellipsoid under Deep Cut

                g' (x − xc) + β ≤ 0

        Arguments:
            beta (float): [description]

        Returns:
            int: [description]
        """
        try:
            tau = math.sqrt(self._tsq)
        except ValueError:
            print("Warning: tsq is negative: {}".format(self._tsq))
            self._tsq = 0.0
            tau = 0.0

        if tau < beta:
            return CutStatus.NoSoln  # no sol'n
        if beta == 0.0:
            self.calc_cc(tau)
            return CutStatus.Success
        gamma = tau + self._n * beta
        if gamma < 0.0:
            return CutStatus.NoEffect  # no effect, unlikely

        # self._mu = (tau - beta) / gamma * self._halfNminus1
        self._rho = gamma / self._nPlus1
        self._sigma = 2.0 * self._rho / (tau + beta)
        self._delta = self._c1 * (1.0 - beta * (beta / self._tsq))
        return CutStatus.Success

    def calc_cc(self, tau: float):
        """Calculate new ellipsoid under Central Cut

        Arguments:
            tau (float): [description]
        """
        # self._mu = self._halfNminus1
        self._sigma = self._c2
        self._rho = tau / self._nPlus1
        self._delta = self._c1


class ell1d:
    __slots__ = ("_r", "_xc")

    def __init__(self, Interval):
        """[summary]

        Arguments:
            I ([type]): [description]
        """
        l, u = Interval
        self._r = (u - l) / 2
        self._xc = l + self._r

    def copy(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        E = ell1d([self._xc - self._r, self._xc + self._r])
        return E

    @property
    def xc(self):
        """[summary]

        Returns:
            float: [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x):
        """[summary]

        Arguments:
            x (float): [description]
        """
        self._xc = x

    def update(self, cut):
        """Update ellipsoid core function using the cut
                g' * (x - xc) + beta <= 0

        Arguments:
            g (floay): cut
            beta (array or scalar): [description]

        Returns:
            status: 0: success
            tau: "volumn" of ellipsoid
        """
        g, beta = cut
        # TODO handle g == 0
        tau = abs(self._r * g)
        tsq = tau**2
        if beta == 0:
            self._r /= 2
            self._xc += -self._r if g > 0 else self._r
            return CutStatus.Success, tsq
        if beta > tau:
            return CutStatus.NoSoln, tsq  # no sol'n
        if beta < -tau:  # unlikely
            return CutStatus.NoEffect, tsq  # no effect

        bound = self._xc - beta / g
        upper = bound if g > 0 else self._xc + self._r
        lower = self._xc - self._r if g > 0 else bound
        self._r = (upper - lower) / 2
        self._xc = lower + self._r
        return CutStatus.Success, tsq
