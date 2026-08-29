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

from typing import Callable, Tuple, Union

import numpy as np

from .ell_base import EllBase
from .ell_config import CutStatus

# Type aliases for better code readability
Mat = np.ndarray
CutChoice = Union[float, np.ndarray]  # single or parallel cut
Cut = Tuple[np.ndarray, CutChoice]  # A cut consists of a gradient and a beta value


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

    def _update_core(self, cut: Cut, cut_strategy: Callable) -> CutStatus:
        r"""Update the ellipsoid by applying a cutting plane.

        Given a gradient :math:`\mathbf{g}` and offset :math:`\beta`,
        the ellipsoid :math:`\{ \mathbf{x} : (\mathbf{x} -
        \mathbf{x}_c)^T \mathbf{M}^{-1} (\mathbf{x} - \mathbf{x}_c)
        \le \kappa^2 \}` is updated as follows:

        .. math::

           \tilde{\mathbf{g}} &= \mathbf{M}\,\mathbf{g} \\[4pt]
           \omega &= \mathbf{g}^T \tilde{\mathbf{g}} \\[4pt]
           \tau^2 &= \kappa\,\omega \\[4pt]
           \mathbf{x}_c &\leftarrow \mathbf{x}_c -
                        \frac{\rho}{\omega}\,\tilde{\mathbf{g}} \\[4pt]
           \mathbf{M} &\leftarrow \mathbf{M} -
                       \frac{\sigma}{\omega}\,
                       \tilde{\mathbf{g}} \tilde{\mathbf{g}}^T \\[4pt]
           \kappa &\leftarrow \kappa \cdot \delta

        where :math:`\rho, \sigma, \delta` are returned by the cut
        strategy (see :class:`~ellalgo.ell_calc_core.EllCalcCore`).

        Args:
            cut: Tuple :math:`(\mathbf{g}, \beta)` for the cut
            cut_strategy: Strategy function to compute :math:`\rho,\sigma,\delta`

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> import numpy as np
            >>> from ellalgo.ell import Ell
            >>> from ellalgo.ell_config import CutStatus
            >>> ell = Ell(1.0, np.array([0.0, 0.0]))
            >>> cut = (np.array([1.0, 1.0]), 0.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel)
            >>> status == CutStatus.Success
            True
        """
        grad, beta = cut
        if np.all(grad == 0.0):
            raise ValueError("Gradient cannot be a zero vector.")
        # Calculate M * grad (matrix-vector multiplication)
        grad_t = self._mq @ grad  # n^2 multiplications
        # Calculate grad^T * (M * grad)
        omega = grad.dot(grad_t)  # n multiplications
        if omega == 0.0:
            return CutStatus.NoEffect
        # Guard against denormal omega that would overflow when
        # computing sigma/omega in the rank-1 update below
        if not (omega > np.finfo(float).tiny):
            return CutStatus.NoEffect
        # Update tsq measure
        self._tsq = self._kappa * omega

        # Get update parameters from the strategy
        status, result = cut_strategy(beta, self._tsq)

        if result is None:
            return status

        # Extract update parameters
        rho, sigma, delta = result

        # Update center point: xc -= (rho/omega) * grad_t
        self._xc -= (rho / omega) * grad_t
        # Update matrix: M -= (sigma/omega) * grad_t * grad_t^T
        self._mq -= (sigma / omega) * np.outer(grad_t, grad_t)
        # Update scaling factor
        self._kappa *= delta

        # Optional: apply scaling immediately rather than deferring
        if self.no_defer_trick:
            self._mq *= self._kappa
            self._kappa = 1.0
        return status
