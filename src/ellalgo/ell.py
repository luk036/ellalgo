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
    """Ellipsoid

    Args:
        SearchSpace (_type_): _description_
        SearchSpaceQ (_type_): _description_

    Returns:
        _type_: _description_
    """
    no_defer_trick: bool = False

    _mq: Mat
    _xc: ArrayType
    _kappa: float
    _tsq: float
    _helper: EllCalc

    def __init__(self, val, xc: ArrayType) -> None:
        """_summary_

        Args:
            val (_type_): _description_
            xc (ArrayType): _description_
        """
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
        """_summary_

        Returns:
            ArrayType: _description_
        """
        return self._xc

    def set_xc(self, xc: ArrayType) -> None:
        """_summary_

        Args:
            xc (ArrayType): _description_
        """
        self._xc = xc

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
        """
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
