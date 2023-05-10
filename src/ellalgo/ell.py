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
    _tsq: float

    def __init__(self, val, xc: ArrayType, CalcStrategy=EllCalc) -> None:
        ndim = len(xc)
        self._helper = CalcStrategy(ndim)
        self._xc = xc
        self._tsq = 0.0
        if isinstance(val, (int, float)):
            self._kappa = val
            self._mq = np.eye(ndim)
        else:
            self._kappa = 1.0
            self._mq = np.diag(val)

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
        return self._tsq

    def update_dc(self, cut) -> CutStatus:
        return self._update_core(cut, self._helper.calc_single_or_ll)

    def update_cc(self, cut) -> CutStatus:
        return self._update_core(cut, self._helper.calc_single_or_ll_cc)

    def update_q(self, cut) -> CutStatus:
        return self._update_core(cut, self._helper.calc_single_or_ll_q)

    def _update_core(self, cut, dc_or_cc_strategy) -> CutStatus:
        grad, beta = cut
        grad_t = self._mq @ grad  # n^2 multiplications
        omega = grad.dot(grad_t)  # n multiplications
        self._tsq = self._kappa * omega

        status, rho, sigma, delta = dc_or_cc_strategy(beta, self._tsq)
        # if central_cut:
        #     status = self._helper.calc_single_or_ll_cc(beta)
        # else:
        #     status = self._helper.calc_single_or_ll(beta)

        if status != CutStatus.Success:
            return status

        self._xc -= (rho / omega) * grad_t
        self._mq -= (sigma / omega) * np.outer(grad_t, grad_t)
        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
