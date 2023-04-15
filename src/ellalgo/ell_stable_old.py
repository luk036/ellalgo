# -*- coding: utf-8 -*-
import math
from typing import Tuple, Union

import numpy as np

from .cutting_plane import CutStatus

Arr = Union[np.ndarray]


class EllStable:
    """Ellipsoid Search Space

        EllStable = {x | (x − xc)' Q^−1 (x − xc) ≤ κ}

    Returns:
        [type] -- [description]

    """

    # __slots__ = ('_n', '_c1', '_kappa', '_rho', '_sigma', '_delta', '_tsq',
    #              '_xc', '_Q', 'use_parallel_cut', 'no_defer_trick')

    def __init__(self, val: Union[Arr, float], x: Arr):
        """Construct a new EllStable object

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
            EllStable: [description]
        """
        ellip = EllStable(self._kappa, self.xc)
        ellip._Q = self._Q.copy()
        # ellip._c1 = self._c1
        ellip.use_parallel_cut = self.use_parallel_cut
        ellip.no_defer_trick = self.no_defer_trick
        return ellip

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

    # Reference: Gill, Murray, and Wright, "Practical Optimization", p43.
    # Author: Brian Borchers (borchers@nmt.edu)
    def update(self, cut) -> Tuple[int, float]:
        g, beta = cut

        # calculate inv(L)*g: (n-1)*n/2 multiplications
        invLg = g.copy()  # initially
        for i in range(1, self._n):
            for j in range(i):
                self._Q[i, j] = self._Q[j, i] * invLg[j]
                # keep for rank-one update
                invLg[i] -= self._Q[i, j]

        # calculate inv(D)*inv(L)*g: n
        invDinvLg = invLg.copy()  # initially
        for i in range(self._n):
            invDinvLg[i] *= self._Q[i, i]

        # calculate omega: n
        gQg = invDinvLg * invLg
        omega = sum(gQg)

        self._tsq = self._kappa * omega

        status = self.calc_ll(beta)
        if status != CutStatus.Success:
            return status, self._tsq

        # calculate Q*g = inv(L')*inv(D)*inv(L)*g : (n-1)*n/2
        Qg = invDinvLg.copy()  # initially
        for i in range(self._n - 1, 0, -1):
            for j in range(i, self._n):
                Qg[i - 1] -= self._Q[i, j] * Qg[j]  # TODO

        # calculate xc: n
        self._xc -= (self._rho / omega) * Qg

        # rank-one update: 3*n + (n-1)*n/2
        # r = self._sigma / omega
        mu = self._sigma / (1.0 - self._sigma)
        oldt = omega / mu  # initially
        m = self._n - 1
        for j in range(m):
            # p=sqrt(k)*vv(j)
            # p = invLg[j]
            # mup = mu * p
            t = oldt + gQg[j]
            # self._Q[j, j] /= t # update invD
            beta2 = invDinvLg[j] / t
            self._Q[j, j] *= oldt / t  # update invD
            for k in range(j + 1, self._n):
                # v(k) -= p * self._Q[j, k]
                self._Q[j, k] += beta2 * self._Q[k, j]
            oldt = t

        # p = invLg(n1)
        # mup = mu * p
        t = oldt + gQg[m]
        self._Q[m, m] *= oldt / t  # update invD
        self._kappa *= self._delta

        # if (self.no_defer_trick)
        # {
        #     self._Q *= self._kappa
        #     self._kappa = 1.
        # }
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
        b1sqn = b1 * (b1 / self._tsq)
        t1n = 1.0 - b1sqn
        if t1n < 0.0 or not self.use_parallel_cut:
            return self.calc_dc(b0)
        if b1 - b0 < 0.0:
            return CutStatus.NoSoln  # no sol'n
        if b0 == 0.0:
            self.calc_ll_cc(b1, b1sqn)
            return CutStatus.Success

        b0b1n = b0 * (b1 / self._tsq)
        if self._n * b0b1n < -1:  # unlikely
            return CutStatus.NoEffect  # no effect

        # parallel cut
        t0n = 1.0 - b0 * (b0 / self._tsq)
        # t1 = self._tsq - b1sq
        bsum = b0 + b1
        bsumn = bsum / self._tsq
        bav = bsum / 2.0
        tempn = self._halfN * bsumn * (b1 - b0)
        xi = math.sqrt(t0n * t1n + tempn * tempn)
        self._sigma = self._c3 + (1.0 - b0b1n - xi) / (bsumn * bav * self._nPlus1)
        self._rho = self._sigma * bav
        self._delta = self._c1 * ((t0n + t1n) / 2.0 + xi / self._n)
        return CutStatus.Success

    def calc_ll_cc(self, b1: float, b1sqn: float):
        """Calculate new ellipsoid under Parallel Cut, one of them is central

                g' (x − xc) ≤ 0
                g' (x − xc) + β1 ≥ 0

        Arguments:
            b1 (float): [description]
            b1sq (float): [description]
        """
        n = self._n
        xi = math.sqrt(1 - b1sqn + (self._halfN * b1sqn) ** 2)
        self._sigma = self._c3 + self._c2 * (1.0 - xi) / b1sqn
        self._rho = self._sigma * b1 / 2.0
        self._delta = self._c1 * (1.0 - b1sqn / 2.0 + xi / n)

    def calc_dc(self, beta: float) -> CutStatus:
        """Calculate new ellipsoid under Deep Cut

                g' (x − xc​) + β ​≤ 0

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

        bdiff = tau - beta
        if bdiff < 0.0:
            return CutStatus.NoSoln  # no sol'n
        if beta == 0.0:
            self.calc_cc(tau)
            return CutStatus.Success
        n = self._n
        gamma = tau + n * beta
        if gamma < 0.0:
            return CutStatus.NoEffect  # no effect, unlikely

        self._mu = (bdiff / gamma) * self._halfNminus1
        self._rho = gamma / self._nPlus1
        self._sigma = 2.0 * self._rho / (tau + beta)
        self._delta = self._c1 * (1.0 - beta * (beta / self._tsq))
        return CutStatus.Success

    def calc_cc(self, tau: float):
        """Calculate new ellipsoid under Central Cut

        Arguments:
            tau (float): [description]
        """
        self._mu = self._halfNminus1
        self._sigma = self._c2
        self._rho = tau / self._nPlus1
        self._delta = self._c1
