from typing import Tuple, Union

import numpy as np

from .cutting_plane import CutStatus, SearchSpace
from .ell_calc import EllCalc

Arr = Union[np.ndarray]
Mat = Union[np.ndarray]
CutChoice = Union[float, Arr]  # single or parallel
Cut = Tuple[Arr, CutChoice]


class Ell(SearchSpace):
    no_defer_trick: bool = False

    _mq: Mat
    _xc: Arr
    _kappa: float
    _helper: EllCalc

    def _new_with_matrix(self, kappa: float, mq: Mat, xc: Arr) -> None:
        n = len(xc)
        self._helper = EllCalc(float(n))
        self._kappa = kappa
        self._mq = mq
        self._xc = xc

    def __init__(self, val, xc: Arr) -> None:
        if np.isscalar(val):
            self._new_with_matrix(val, np.eye(len(xc)), xc)
        else:
            self._new_with_matrix(1.0, np.diag(val), xc)

    def copy(self):
        """[summary]

        Returns:
            Ell: [description]
        """
        E = Ell(self._kappa, self.xc)
        E._mq = self._mq.copy()
        E._helper = self._helper.copy()
        E.no_defer_trick = self.no_defer_trick
        return E

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

        Arguments:
            x ([type]): [description]
        """
        self._xc = x

    def update(self, cut: Cut) -> Tuple[CutStatus, float]:
        grad, beta = cut
        grad_t = self._mq @ grad  # n^2 multiplications
        omega = grad @ grad_t  # n multiplications
        self._helper.tsq = self._kappa * omega

        status = self._helper.calc_ll(beta)
        if status != CutStatus.Success:
            return (status, self._helper.tsq)

        self._xc -= (self._helper.rho / omega) * grad_t
        self._mq -= (self._helper.sigma / omega) * np.outer(grad_t, grad_t)
        self._kappa *= self._helper.delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return (status, self._helper.tsq)
