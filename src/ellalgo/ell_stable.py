from typing import Tuple, Union

import numpy as np

from .cutting_plane import CutStatus
from .ell_calc import EllCalc

Arr = Union[np.ndarray, float]
Mat = Union[np.ndarray, float]


class EllStable:
    no_defer_trick: bool = False

    _mq: Mat
    _xc: Arr
    _kappa: float
    _helper: EllCalc
    _n: int

    def _new_with_matrix(self, kappa: float, mq: Mat, xc: Arr):
        n = len(xc)
        self._helper = EllCalc(float(n))
        self._kappa = kappa
        self._mq = mq
        self._xc = xc
        self._n = n

    def __init__(self, val, xc: Arr):
        if np.isscalar(val):
            self._new_with_matrix(val, np.eye(len(xc)), xc)
        else:
            self._new_with_matrix(1.0, np.diag(val), xc)

    def copy(self):
        """[summary]

        Returns:
            EllStable: [description]
        """
        E = EllStable(self._kappa, self.xc)
        E._mq = self._mq.copy()
        E._helper = self._helper.copy()
        E._n = self._n
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

        arguments:
            x ([type]): [description]
        """
        self._xc = x

    def update(self, cut) -> Tuple[CutStatus, float]:
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
        for i in range(1, self._n):
            for j in range(i):
                self._mq[i, j] = self._mq[j, i] * invLg[j]
                # keep for rank-one update
                invLg[i] -= self._mq[i, j]

        # calculate inv(D)*inv(L)*g: n
        invDinvLg = invLg.copy()  # initially
        for i in range(self._n):
            invDinvLg[i] *= self._mq[i, i]

        # calculate omega: n
        gg_t = invDinvLg * invLg
        omega = sum(gg_t)

        self._helper.tsq = self._kappa * omega
        status = self._helper.calc_ll(beta)
        if status != CutStatus.Success:
            return (status, self._helper.tsq)

        # calculate Q*g = inv(L')*inv(D)*inv(L)*g : (n-1)*n/2
        g_t = invDinvLg.copy()  # initially
        for i in range(self._n - 1, 0, -1):
            for j in range(i, self._n):
                g_t[i - 1] -= self._mq[i, j] * g_t[j]  # TODO

        # calculate xc: n
        self._xc -= (self._helper.rho / omega) * g_t

        # rank-one update: 3*n + (n-1)*n/2
        # r = self._sigma / omega
        mu = self._helper.sigma / (1.0 - self._helper.sigma)
        oldt = omega / mu  # initially
        m = self._n - 1
        for j in range(m):
            # p=sqrt(k)*vv(j)
            # p = invLg[j]
            # mup = mu * p
            t = oldt + gg_t[j]
            # self._mq[j, j] /= t # update invD
            beta2 = invDinvLg[j] / t
            self._mq[j, j] *= oldt / t  # update invD
            for k in range(j + 1, self._n):
                # v(k) -= p * self._mq[j, k]
                self._mq[j, k] += beta2 * self._mq[k, j]
            oldt = t

        # p = invLg(n1)
        # mup = mu * p
        t = oldt + gg_t[m]
        self._mq[m, m] *= oldt / t  # update invD
        self._kappa *= self._helper.delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return (status, self._helper.tsq)
