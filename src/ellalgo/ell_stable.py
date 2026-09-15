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

from typing import Tuple, Union

import numpy as np

from .ell_base import EllBase
from .ell_config import CutStatus

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

    def _update_core(self, cut: Cut, cut_strategy) -> CutStatus:
        r"""Update the ellipsoid using :math:`LDL^T` factorization.

        The shape matrix is stored as :math:`\mathbf{M} = \kappa \mathbf{LDL}^T`.
        Forward/backward substitution replaces explicit matrix–vector products.

        Uses pre-allocated scratch buffers (Rust-style) to eliminate per-call
        memory allocation. See :class:`EllStable` for buffer documentation.

        :param cut: Tuple :math:`(\mathbf{g}, \beta)` for the cut
        :param cut_strategy: Strategy function to compute :math:`\rho,\sigma,\delta`
        :return: A :class:`CutStatus` object

        Examples:
            >>> import numpy as np
            >>> from ellalgo.ell_stable import EllStable
            >>> ell = EllStable(1.0, np.array([0.0, 0.0]))
            >>> cut = (np.array([1.0, 1.0]), 1.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel)
            >>> status == CutStatus.Success
            True
        """
        g, beta = cut

        # --- forward substitution: w = L^{-1} * g — reuse _inv_lower_g ---
        np.copyto(self._inv_lower_g, g)
        for j in range(self._ndim - 1):
            col = self._mq[j + 1 :, j] * self._inv_lower_g[j]
            self._mq[j, j + 1 :] = col
            self._inv_lower_g[j + 1 :] -= col

        # --- z = D^{-1} * w — reuse _inv_diag_inv_lower_g ---
        np.copyto(self._inv_diag_inv_lower_g, self._inv_lower_g)
        self._inv_diag_inv_lower_g *= np.diagonal(self._mq)

        # --- omega = sum(w_i * z_i) — no gg_t buffer needed ---
        omega = float(self._inv_lower_g @ self._inv_diag_inv_lower_g)

        self._tsq = self._kappa * omega

        status, result = cut_strategy(beta, self._tsq)
        if result is None:
            return status

        rho, sigma, delta = result

        # --- back substitution: q = L^{-T} * z — reuse _g_t ---
        np.copyto(self._g_t, self._inv_diag_inv_lower_g)
        for i in range(self._ndim - 1, 0, -1):
            self._g_t[i - 1] -= self._mq[i:, i - 1] @ self._g_t[i:]

        # --- center update ---
        self._xc -= (rho / omega) * self._g_t

        # --- rank-one LDL^T update — reuse _g_t as working vector v ---
        mu = sigma / (1.0 - sigma)
        if mu == 0.0:
            return status
        oldt = omega / mu
        np.copyto(self._g_t, g)  # v = gradient (g_t buffer no longer needed as q)
        for j in range(self._ndim):
            temp = self._inv_diag_inv_lower_g[j]
            newt = oldt + self._g_t[j] * temp
            beta2 = temp / newt
            self._mq[j, j] *= oldt / newt
            self._g_t[j + 1 :] -= self._mq[j, j + 1 :]
            self._mq[j + 1 :, j] += beta2 * self._g_t[j + 1 :]
            oldt = newt

        self._kappa *= delta

        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
