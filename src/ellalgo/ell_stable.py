"""
Numerically stable ellipsoid implementation using LDL^T factorization.

The `EllStable` class implements the ellipsoid method using a square-root-free
Cholesky (LDL^T) decomposition of the shape matrix. This approach avoids
explicit matrix inversion and provides better numerical stability, especially
for ill-conditioned problems.

Key differences from `Ell`:
    - Stores the LDL^T factors of the shape matrix directly
    - Uses forward/backward substitution instead of matrix-vector products
    - Implements rank-one updates for the LDL^T factors
"""

from typing import Callable, Tuple, Union

import numpy as np

from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import ArrayType, SearchSpace, SingleCut

Matrix = np.ndarray
CutChoice = Union[SingleCut, ArrayType]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]


class EllStable(SearchSpace[ArrayType]):
    """Numerically stable ellipsoid search space using LDL^T factorization.

    This class stores the ellipsoid's shape matrix in LDL^T factored form and
    performs rank-one updates directly on the factors, avoiding the need for
    explicit matrix inversion. This provides better numerical stability for
    ill-conditioned problems.

    Examples:
        >>> import numpy as np
        >>> from ellalgo.ell_stable import EllStable
        >>> ell = EllStable(1.0, np.array([0.0, 0.0]))
        >>> ell.xc()
        array([0., 0.])
        >>> ell.tsq()
        0.0
    """

    no_defer_trick: bool = False

    _mq: Matrix
    _xc: ArrayType
    _kappa: float
    _tsq: float
    _ndim: int
    helper: EllCalc
    # Pre-allocated scratch buffers (match Rust's strategy: zero per-call allocation)
    _inv_lower_g: ArrayType  # w = L^{-1}g (forward substitution)
    _inv_diag_inv_lower_g: ArrayType  # z = D^{-1}w
    _g_t: ArrayType  # q = L^{-T}z (back substitution), then v (rank-1 update)

    def __init__(self, val: Union[float, ArrayType], x_center: ArrayType) -> None:
        ndim = len(x_center)
        self.helper = EllCalc(ndim)
        self._xc = x_center
        self._tsq = 0.0
        self._ndim = ndim
        # Pre-allocate scratch buffers (Rust-style: avoid per-call allocation)
        self._inv_lower_g = np.empty(ndim)
        self._inv_diag_inv_lower_g = np.empty(ndim)
        self._g_t = np.empty(ndim)
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

    def set_xc(self, x_center: ArrayType) -> None:
        """
        The function sets the value of the variable `_xc` to the input `x_center`.

        :param x_center: The parameter `x_center` is of type `ArrayType`
        :type x_center: ArrayType
        """
        self._xc = x_center

    def tsq(self) -> float:
        """
        The function `tsq` returns the measure of the distance between `xc` and `x*`.
        :return: The method is returning a float value, which represents the measure of the distance between xc and x*.
        """
        return self._tsq

    def update_bias_cut(self, cut: Cut) -> CutStatus:
        """
        The function `update_bias_cut` is an implementation of the `SearchSpace` interface that updates the
        cut status based on a given cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents some kind of cut
        :return: a `CutStatus` object.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell.update_bias_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel)

    def update_central_cut(self, cut: Cut) -> CutStatus:
        """
        The function `update_central_cut` is an implementation of the `SearchSpace` interface that updates the
        cut status based on a given cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents a cut
        :return: a `CutStatus` object.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 0.0)
            >>> status = ell.update_central_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_central_cut)

    def update_q(self, cut: Cut) -> CutStatus:
        """
        The function `update_q` is an implementation of the `SearchSpaceQ` interface that updates the
        cut status based on a given cut.

        :param cut: The `cut` parameter is of type `_type_` and it represents the cut that needs to be
            updated
        :return: a `CutStatus` object.

        Examples:
            >>> ell = EllStable(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), -0.01)
            >>> status = ell.update_q(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_q)

    # private:

    def _update_core(self, cut: Cut, cut_strategy: Callable) -> CutStatus:
        r"""Update the ellipsoid using :math:`LDL^T` factorization.

        The shape matrix is stored as :math:`\mathbf{M} = \kappa \mathbf{LDL}^T`.
        Forward/backward substitution replaces explicit matrix–vector products.

        Uses pre-allocated scratch buffers (Rust-style) to eliminate per-call
        memory allocation. See :class:`EllStable` for buffer documentation.

        :param cut: Tuple :math:`(\mathbf{g}, \beta)` for the cut
        :param cut_strategy: Strategy function to compute :math:`\rho,\sigma,\delta`
        :return: A :class:`CutStatus` object
        """
        g, beta = cut

        # --- forward substitution: w = L^{-1} * g — reuse _inv_lower_g ---
        np.copyto(self._inv_lower_g, g)
        for j in range(self._ndim - 1):
            for i in range(j + 1, self._ndim):
                self._mq[j, i] = self._mq[i, j] * self._inv_lower_g[j]
                self._inv_lower_g[i] -= self._mq[j, i]

        # --- z = D^{-1} * w — reuse _inv_diag_inv_lower_g ---
        np.copyto(self._inv_diag_inv_lower_g, self._inv_lower_g)
        for i in range(self._ndim):
            self._inv_diag_inv_lower_g[i] *= self._mq[i, i]

        # --- omega = sum(w_i * z_i) — no gg_t buffer needed ---
        omega = 0.0
        for i in range(self._ndim):
            omega += self._inv_lower_g[i] * self._inv_diag_inv_lower_g[i]

        self._tsq = self._kappa * omega

        status, result = cut_strategy(beta, self._tsq)
        if result is None:
            return status

        rho, sigma, delta = result

        # --- back substitution: q = L^{-T} * z — reuse _g_t ---
        np.copyto(self._g_t, self._inv_diag_inv_lower_g)
        for i in range(self._ndim - 1, 0, -1):
            for j in range(i, self._ndim):
                self._g_t[i - 1] -= self._mq[j, i - 1] * self._g_t[j]

        # --- center update ---
        self._xc -= (rho / omega) * self._g_t

        # --- rank-one LDL^T update — reuse _g_t as working vector v ---
        mu = sigma / (1.0 - sigma)
        if mu == 0.0:
            return status
        oldt = omega / mu
        np.copyto(self._g_t, g)  # v = gradient (g_t buffer no longer needed as q)
        for j in range(self._ndim):
            p = self._g_t[j]
            temp = self._inv_diag_inv_lower_g[j]
            newt = oldt + p * temp
            beta2 = temp / newt
            self._mq[j, j] *= oldt / newt
            for k in range(j + 1, self._ndim):
                self._g_t[k] -= self._mq[j, k]
                self._mq[k, j] += beta2 * self._g_t[k]
            oldt = newt

        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
