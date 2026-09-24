"""
Ellipsoid search space implementation for the ellipsoid method.

The `Ell` class represents a convex search space as an n-dimensional ellipsoid
defined by a center point and a shape matrix. It provides methods to update the
ellipsoid when cutting planes (deep cuts, central cuts, parallel cuts) are applied,
progressively narrowing the search region toward an optimal solution.

This module implements the classic direct Q-update strategy. See
:class:`EllBase` for the shared public API.

Key operations:
    - update_bias_cut: Apply a deep (non-central) cut
    - update_central_cut: Apply a central cut through the ellipsoid center
    - update_q: Apply a cut for discrete/quantized optimization
"""

from typing import Optional, Tuple, Union

import numpy as np

from .ell_base import EllBase

# Type aliases for better code readability
Mat = np.ndarray
CutChoice = Union[float, np.ndarray]  # single or parallel cut
Cut = Tuple[np.ndarray, CutChoice]  # A cut consists of a gradient and a beta value

_TINY = float(np.finfo(np.float64).tiny)


class Ell(EllBase[np.ndarray]):
    """Ellipsoid Search Space (classic direct Q-update strategy).

    Concrete strategy of :class:`EllBase`: updates the shape matrix directly
    via the rank-1 update
    :math:`M \\leftarrow M - (\\sigma/\\omega)\\,\\tilde{g}\\tilde{g}^T`.

    Examples:
        >>> import numpy as np
        >>> from ellalgo.ell import Ell
        >>> ell = Ell(1.0, np.array([0.0, 0.0]))
        >>> ell.xc()
        array([0., 0.])
        >>> ell.tsq()
        0.0
    """

    # private:

    def _omega(self, g: np.ndarray) -> Optional[Tuple[float, Optional[np.ndarray]]]:
        r"""Compute :math:`\omega` and the cached matrix-vector product.

        .. math::

           \tilde{\mathbf{g}} &= \mathbf{M}\,\mathbf{g} \\[4pt]
           \omega &= \mathbf{g}^T \tilde{\mathbf{g}}

        Returns ``None`` when :math:`\omega` is zero or denormal (so the cut
        has no effect); otherwise returns ``(omega, \tilde{g})``.
        """
        if not g.any():
            raise ValueError("Gradient cannot be a zero vector.")
        g_t = self._mq @ g  # n^2 multiplications
        omega = g.dot(g_t)  # n multiplications
        if omega == 0.0:
            return None
        # Guard against denormal omega that would overflow when
        # computing sigma/omega in the rank-1 update below
        if not (omega > _TINY):
            return None
        return omega, g_t

    def _apply_update(
        self,
        g: np.ndarray,
        omega: float,
        g_t: Optional[np.ndarray],
        rho: float,
        sigma: float,
        delta: float,
    ) -> bool:
        r"""Apply the rank-1 center and shape-matrix update.

        .. math::

           \mathbf{x}_c &\leftarrow \mathbf{x}_c -
                        \frac{\rho}{\omega}\,\tilde{\mathbf{g}} \\[4pt]
           \mathbf{M} &\leftarrow \mathbf{M} -
                       \frac{\sigma}{\omega}\,
                       \tilde{\mathbf{g}} \tilde{\mathbf{g}}^T
        """
        assert g_t is not None
        self._xc -= (rho / omega) * g_t
        self._mq -= (sigma / omega) * (g_t[:, None] * g_t)
        return True
