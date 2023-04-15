from typing import Tuple, Union

import numpy as np

from .cutting_plane import CutStatus
from .ell_calc import EllCalc

Arr = Union[np.ndarray]
Mat = Union[np.ndarray]
CutChoice = Union[float, Arr]  # single or parallel
Cut = Tuple[Arr, CutChoice]


class EllStable:
    no_defer_trick: bool = False

    _mq: Mat
    _xc: Arr
    _kappa: float
    _helper: EllCalc
    _n: int

    def _new_with_matrix(self, kappa: float, mq: Mat, xc: Arr) -> None:
        n = len(xc)
        self._helper = EllCalc(n)
        self._kappa = kappa
        self._mq = mq
        self._xc = xc
        self._n = n

    def __init__(self, val, xc: Arr) -> None:
        if np.isscalar(val):
            self._new_with_matrix(val, np.eye(len(xc)), xc)
        else:
            self._new_with_matrix(1.0, np.diag(val), xc)

    def copy(self):
        """[summary]

        Returns:
            EllStable: [description]
        """
        ellip = EllStable(self._kappa, self._xc)
        ellip._mq = self._mq.copy()
        ellip._helper = self._helper.copy()
        ellip._n = self._n
        ellip.no_defer_trick = self.no_defer_trick
        return ellip

    # @property
    def xc(self) -> Arr:
        """copy the whole array anyway

        Returns:
            [type]: [description]
        """
        return self._xc

    # @xc.setter
    def set_xc(self, x: Arr) -> None:
        """Set the xc object

        arguments:
            x ([type]): [description]
        """
        self._xc = x

    # @property
    def tsq(self) -> float:
        """Measure the distance square between xc and x*

        Returns:
            [type]: [description]
        """
        return self._helper.tsq

    def update(self, cut: Cut, central_cut: bool = False) -> CutStatus:
        """Update ellipsoid by cut

        Arguments:
            cut ([type]): [description]

        Returns:
            [type]: [description]

        Reference:
            Gill, Murray, and Wright, "Practical Optimization", p43.
        """
        g, beta = cut

        # calculate inv(L)*g: (n-1)*n/2 multiplications
        invLg = g.copy()  # initially

        for j in range(self._n - 1):
            for i in range(j + 1, self._n):
                self._mq[j, i] = self._mq[i, j] * invLg[j]
                # keep for rank-one update
                invLg[i] -= self._mq[j, i]

        # calculate inv(D)*inv(L)*g: n
        invDinvLg = invLg.copy()  # initially
        for i in range(self._n):
            invDinvLg[i] *= self._mq[i, i]

        # print(invDinvLg)
        # calculate omega: n
        gg_t = invLg * invDinvLg
        omega = sum(gg_t)

        self._helper.tsq = self._kappa * omega  # need for helper

        if central_cut:
            status = self._helper.calc_single_or_ll_cc(beta)
        else:
            status = self._helper.calc_single_or_ll(beta)

        if status != CutStatus.Success:
            return status

        # calculate Q*g = inv(L')*inv(D)*inv(L)*g : (n-1)*n/2
        g_t = invDinvLg.copy()  # initially
        # print(g_t)
        # print(self._mq)
        for i in range(self._n - 1, 0, -1):
            for j in range(i, self._n):
                g_t[i - 1] -= self._mq[j, i - 1] * g_t[j]  # TODO

        # print(g_t)
        # calculate xc: n
        self._xc -= (self._helper.rho / omega) * g_t

        # rank-one update: 3*n + (n-1)*n/2
        # r = self._sigma / omega
        mu = self._helper.sigma / (1.0 - self._helper.sigma)
        oldt = omega / mu  # initially
        v = g.copy()
        for j in range(self._n):
            p = v[j]
            # temp = p * self._mq[j, j]
            temp = invDinvLg[j]
            newt = oldt + p * temp
            beta2 = temp / newt
            self._mq[j, j] *= oldt / newt  # update invD
            for k in range(j + 1, self._n):
                # v[k] -= p * self._mq[k, j]
                v[k] -= self._mq[j, k]
                self._mq[k, j] += beta2 * v[k]
            oldt = newt

        self._kappa *= self._helper.delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
