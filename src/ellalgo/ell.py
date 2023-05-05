import numpy as np
# from .cutting_plane import CutStatus
from .ell_calc import EllCalc
from .ell_config import CutStatus
from typing import Tuple, Union

Mat = np.ndarray
ArrayType = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]


class Ell:
    no_defer_trick: bool = False

    _mq: Mat
    _xc: ArrayType
    _kappa: float
    _helper: EllCalc

    def _new_with_matrix(self, kappa, mq: Mat, xc: ArrayType) -> None:
        n = len(xc)
        self._helper = EllCalc(n)
        self._kappa = kappa
        self._mq = mq
        self._xc = xc

    def __init__(self, val, xc: ArrayType) -> None:
        if np.isscalar(val):
            self._new_with_matrix(val, np.eye(len(xc)), xc)
        else:
            self._new_with_matrix(1.0, np.diag(val), xc)

    def copy(self):
        """[summary]

        Returns:
            Ell: [description]
        """
        ellip = Ell(self._kappa, self._xc)
        ellip._mq = self._mq.copy()
        ellip._helper = self._helper.copy()
        ellip.no_defer_trick = self.no_defer_trick
        return ellip

    # @property
    def xc(self) -> ArrayType:
        """copy the whole array anyway

        Returns:
            [type]: [description]
        """
        return self._xc

    # @xc.setter
    def set_xc(self, x: ArrayType) -> None:
        """Set the xc object

        Arguments:
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

    def update(self, cut, central_cut: bool = False) -> CutStatus:
        grad, beta = cut
        grad_t = self._mq @ grad  # n^2 multiplications
        omega = grad.dot(grad_t)  # n multiplications
        self._helper.tsq = self._kappa * omega

        if central_cut:
            status = self._helper.calc_single_or_ll_cc(beta)
        else:
            status = self._helper.calc_single_or_ll(beta)

        if status != CutStatus.Success:
            return status

        self._xc -= (self._helper.rho / omega) * grad_t
        self._mq -= (self._helper.sigma / omega) * np.outer(grad_t, grad_t)
        self._kappa *= self._helper.delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
