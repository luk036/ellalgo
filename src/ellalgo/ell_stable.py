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

See :class:`EllBase` for the shared public API.
"""

from typing import Optional, Tuple, Union

import numpy as np

from .ell_base import EllBase

Matrix = np.ndarray
CutChoice = Union[float, np.ndarray]  # single or parallel
Cut = Tuple[np.ndarray, CutChoice]


class EllStable(EllBase[np.ndarray]):
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

    _ndim: int
    # Pre-allocated scratch buffers (match Rust's strategy: zero per-call allocation)
    _inv_lower_g: np.ndarray  # w = L^{-1}g (forward substitution)
    _inv_diag_inv_lower_g: np.ndarray  # z = D^{-1}w
    _g_t: np.ndarray  # q = L^{-T}z (back substitution), then v (rank-1 update)

    def __init__(self, val: Union[float, np.ndarray], x_center: np.ndarray) -> None:
        ndim = len(x_center)
        super().__init__(val, x_center)
        self._ndim = ndim
        # Pre-allocate scratch buffers (Rust-style: avoid per-call allocation)
        self._inv_lower_g = np.empty(ndim)
        self._inv_diag_inv_lower_g = np.empty(ndim)
        self._g_t = np.empty(ndim)

    # private:

    def _omega(self, g: np.ndarray) -> Optional[Tuple[float, Optional[np.ndarray]]]:
        r"""Compute :math:`\omega` by forward substitution.

        Solves :math:`\mathbf{w} = \mathbf{L}^{-1}\mathbf{g}` and
        :math:`\mathbf{z} = \mathbf{D}^{-1}\mathbf{w}`, then
        :math:`\omega = \mathbf{w}^T\mathbf{z}`. The factors of
        :math:`\mathbf{M} = \kappa\mathbf{LDL}^T` are updated in place using
        the pre-allocated scratch buffers.
        """
        np.copyto(self._inv_lower_g, g)
        for j in range(self._ndim - 1):
            col = self._mq[j + 1 :, j] * self._inv_lower_g[j]
            self._mq[j, j + 1 :] = col
            self._inv_lower_g[j + 1 :] -= col
        np.copyto(self._inv_diag_inv_lower_g, self._inv_lower_g)
        self._inv_diag_inv_lower_g *= np.diagonal(self._mq)
        omega = float(self._inv_lower_g @ self._inv_diag_inv_lower_g)
        return omega, None

    def _apply_update(
        self,
        g: np.ndarray,
        omega: float,
        g_t: Optional[np.ndarray],
        rho: float,
        sigma: float,
        delta: float,
    ) -> bool:
        r"""Apply back-substitution, center move, and the rank-one LDL^T update.

        Solves :math:`\mathbf{q} = \mathbf{L}^{-T}\mathbf{z}`, moves the center
        by :math:`-(\rho/\omega)\mathbf{q}`, then applies the rank-one
        :math:`LDL^T` update in place. Returns ``False`` (skipping the scale
        update) when :math:`\mu = \sigma/(1-\sigma)` is zero.
        """
        np.copyto(self._g_t, self._inv_diag_inv_lower_g)
        for i in range(self._ndim - 1, 0, -1):
            self._g_t[i - 1] -= self._mq[i:, i - 1] @ self._g_t[i:]
        self._xc -= (rho / omega) * self._g_t
        mu = sigma / (1.0 - sigma)
        if mu == 0.0:
            return False
        oldt = omega / mu
        np.copyto(self._g_t, g)
        for j in range(self._ndim):
            temp = self._inv_diag_inv_lower_g[j]
            newt = oldt + self._g_t[j] * temp
            beta2 = temp / newt
            self._mq[j, j] *= oldt / newt
            self._g_t[j + 1 :] -= self._mq[j, j + 1 :]
            self._mq[j + 1 :, j] += beta2 * self._g_t[j + 1 :]
            oldt = newt
        return True
