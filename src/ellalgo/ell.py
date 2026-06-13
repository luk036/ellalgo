"""
Ellipsoid search space implementation for the ellipsoid method.

The `Ell` class represents a convex search space as an n-dimensional ellipsoid
defined by a center point and a shape matrix. It provides methods to update the
ellipsoid when cutting planes (deep cuts, central cuts, parallel cuts) are applied,
progressively narrowing the search region toward an optimal solution.

Key operations:
    - update_bias_cut: Apply a deep (non-central) cut
    - update_central_cut: Apply a central cut through the ellipsoid center
    - update_q: Apply a cut for discrete/quantized optimization
"""

from typing import Callable, Tuple, Union

import numpy as np

from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import ArrayType, SearchSpace

# Type aliases for better code readability
Mat = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel cut
Cut = Tuple[ArrayType, CutChoice]  # A cut consists of a gradient and a beta value


# The `Ell` class represents an ellipsoidal search space.
class Ell(SearchSpace[ArrayType]):
    """Ellipsoid Search Space.

    The `Ell` class represents an ellipsoid, which is a search space for convex
    optimization problems. It provides methods to update the ellipsoid based on
    cutting planes, which are used to narrow down the search for an optimal
    solution.
    """

    # Class variable to control whether to defer the matrix scaling trick
    no_defer_trick: bool = False

    # Instance variables:
    _mq: Mat  # Shape matrix
    _xc: ArrayType  # Center point of the ellipsoid
    _kappa: float  # Scaling factor for the ellipsoid matrix
    _tsq: float  # Measure of distance between current center and optimal point
    helper: EllCalc  # Helper object for ellipsoid calculations

    def __init__(self, val: Union[float, ArrayType], x_center: ArrayType) -> None:
        """
        Initialize the ellipsoid with given parameters.

        The initialization can be done in two ways:
        1. With a scalar value (kappa) which creates a unit matrix
        2. With a list of values which creates a diagonal matrix

        Args:
            val: Either a scalar (kappa) or a list of values for diagonal matrix
            x_center: The initial center point of the ellipsoid

        The method:
        1. Determines the dimension from x_center
        2. Creates a helper object for calculations
        3. Sets the center point
        4. Initializes tsq to 0
        5. Sets either kappa with unit matrix or diagonal matrix based on val type

        Examples:
            >>> import numpy as np
            >>> from ellalgo.ell import Ell
            >>> ell = Ell(1.0, np.array([0.0, 0.0]))
            >>> ell.xc()
            array([0., 0.])
        """
        ndim = len(x_center)
        self.helper = EllCalc(ndim)
        self._xc = x_center
        self._tsq = 0.0
        if isinstance(val, (int, float)):
            # Case 1: val is a scalar (kappa), create identity matrix
            self._kappa = val
            self._mq = np.eye(ndim)
        else:
            # Case 2: val is a list/array, create diagonal matrix
            self._kappa = 1.0
            self._mq = np.diag(val)

    def xc(self) -> ArrayType:
        """
        Getter method for the ellipsoid's center point.

        Returns:
            The current center point (_xc) of the ellipsoid
        """
        return self._xc

    def set_xc(self, x_center: ArrayType) -> None:
        """
        Setter method for the ellipsoid's center point.

        Args:
            x_center: The new center point for the ellipsoid
        """
        self._xc = x_center

    def tsq(self) -> float:
        """
        Getter method for the tsq value.

        tsq represents the measure of distance between current center (xc) and optimal point (x*).
        It's calculated as kappa * omega, where omega is grad^T * M * grad.

        Returns:
            The current tsq value
        """
        return self._tsq

    def update_bias_cut(self, cut: Cut) -> CutStatus:
        """
        Update the ellipsoid using a bias cut (deep cut) strategy.

        A bias cut is a general cut that can be either deep or shallow.
        This method delegates to _update_core with the standard cut strategy.

        Args:
            cut: A tuple containing (gradient, beta) for the cut

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> import numpy as np
            >>> from ellalgo.ell import Ell
            >>> from ellalgo.ell_config import CutStatus
            >>> ell = Ell(1.0, np.array([0.0, 0.0]))
            >>> cut = (np.array([1.0, 1.0]), 0.0)
            >>> status = ell.update_bias_cut(cut)
            >>> status == CutStatus.Success
            True
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel)

    def update_central_cut(self, cut: Cut) -> CutStatus:
        """
        Update the ellipsoid using a central cut strategy.

        A central cut is a special case where beta = 0, meaning the cut passes
        exactly through the center of the current ellipsoid.

        Args:
            cut: A tuple containing (gradient, beta) for the cut

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> import numpy as np
            >>> from ellalgo.ell import Ell
            >>> from ellalgo.ell_config import CutStatus
            >>> ell = Ell(1.0, np.array([0.0, 0.0]))
            >>> cut = (np.array([1.0, 1.0]), 0.0)
            >>> status = ell.update_central_cut(cut)
            >>> status == CutStatus.Success
            True
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_central_cut)

    def update_q(self, cut: Cut) -> CutStatus:
        """
        Update the ellipsoid using a non-central cut strategy for Q.

        This is used for non-central cuts (either deep or shallow) in Q space.

        Args:
            cut: A tuple containing (gradient, beta) for the cut

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> import numpy as np
            >>> from ellalgo.ell import Ell
            >>> from ellalgo.ell_config import CutStatus
            >>> ell = Ell(1.0, np.array([0.0, 0.0]))
            >>> cut = (np.array([1.0, 1.0]), 0.01)
            >>> status = ell.update_q(cut)
            >>> status == CutStatus.Success
            True
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_q)

    # private:

    def _update_core(self, cut: Cut, cut_strategy: Callable) -> CutStatus:
        """
        Core method for updating the ellipsoid based on a cut and strategy.

        This method:
        1. Extracts gradient and beta from the cut
        2. Calculates grad_t = M * grad
        3. Computes omega = grad^T * grad_t
        4. Updates tsq = kappa * omega
        5. Uses the cut strategy to get update parameters
        6. Updates the center, matrix, and kappa if successful

        Args:
            cut: A tuple containing (gradient, beta) for the cut
            cut_strategy: The strategy function to calculate update parameters

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
