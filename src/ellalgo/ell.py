import numpy as np
from .cutting_plane import SearchSpace, SearchSpaceQ
from .ell_calc import EllCalc
from .ell_config import CutStatus
from typing import Tuple, Union, Callable

Mat = np.ndarray
ArrayType = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]


class Ell(SearchSpace, SearchSpaceQ):
    no_defer_trick: bool = False

    _mq: Mat
    _xc: ArrayType
    _kappa: float
    _tsq: float
    _helper: EllCalc

    def __init__(self, val, xc: ArrayType) -> None:
        ndim = len(xc)
        self._helper = EllCalc(ndim)
        self._xc = xc
        self._tsq = 0.0
        if isinstance(val, (int, float)):
            self._kappa = val
            self._mq = np.eye(ndim)
        else:
            self._kappa = 1.0
            self._mq = np.diag(val)

    def xc(self) -> ArrayType:
        """copy the whole array anyway

        Returns:
            [type]: [description]
        """
        return self._xc

    def set_xc(self, x: ArrayType) -> None:
        """Set the xc object

        Arguments:
            x ([type]): [description]
        """
        self._xc = x

    def tsq(self) -> float:
        """Measure the distance square between xc and x*

        Returns:
            [type]: [description]
        """
        return self._tsq

    # Implement SearchSpace interface
    def update_dc(self, cut) -> CutStatus:
        return self._update_core(cut, self._helper.calc_single_or_ll)

    # Implement SearchSpace interface
    def update_cc(self, cut) -> CutStatus:
        return self._update_core(cut, self._helper.calc_single_or_ll_cc)

    # Implement SearchSpaceQ interface
    def update_q(self, cut) -> CutStatus:
        return self._update_core(cut, self._helper.calc_single_or_ll_q)

    # private:

    def _update_core(self, cut, cut_strategy: Callable) -> CutStatus:
        grad, beta = cut
        grad_t = self._mq @ grad  # n^2 multiplications
        omega = grad.dot(grad_t)  # n multiplications
        self._tsq = self._kappa * omega

        status, rho, sigma, delta = cut_strategy(beta, self._tsq)

        if status != CutStatus.Success:
            return status

        self._xc -= (rho / omega) * grad_t
        self._mq -= (sigma / omega) * np.outer(grad_t, grad_t)
        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
