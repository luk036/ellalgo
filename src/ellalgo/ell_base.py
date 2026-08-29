"""
Shared ellipsoid search space base (Strategy pattern).

The `EllBase` class implements the common public API of the ellipsoid search
space: construction from either a scalar (kappa) or per-axis values, center
accessors, and the three cutting-plane update entry points. The actual
ellipsoid update (`_update_core`) is delegated to concrete subclasses:

    - `Ell`: classic direct Q-update (see :mod:`ellalgo.ell`)
    - `EllStable`: numerically-stable LDL^T update (see
      :mod:`ellalgo.ell_stable`)

This mirrors the Strategy pattern: the cutting-plane drivers only depend on
the `SearchSpace` protocol (xc, tsq, update_*), and the update strategy is
selected by choosing the concrete subclass.
"""

from typing import Tuple, Union

import numpy as np

from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import ArrayType, SearchSpace, SingleCut

# Type aliases for better code readability
Mat = np.ndarray
CutChoice = Union[SingleCut, ArrayType]  # single or parallel cut
Cut = Tuple[ArrayType, CutChoice]  # A cut consists of a gradient and a beta value


class EllBase(SearchSpace[ArrayType]):
    """Ellipsoid Search Space (shared implementation).

    The `EllBase` class represents an ellipsoid, which is a search space for
    convex optimization problems. It provides methods to update the ellipsoid
    based on cutting planes, which are used to narrow down the search for an
    optimal solution.

    Concrete subclasses implement :meth:`_update_core`, which applies the
    cutting plane to the ellipsoid using a specific matrix update strategy.
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

    @classmethod
    def from_radii(cls, val: ArrayType, x_center: ArrayType) -> "EllBase":
        """Create an ellipsoid from per-axis radii.

        Args:
            val: Per-axis radii (diagonal of the shape matrix).
            x_center: The initial center point of the ellipsoid.

        Returns:
            A new ellipsoid initialized from per-axis values.
        """
        return cls(val, x_center)

    @classmethod
    def from_alpha(cls, alpha: float, x_center: ArrayType) -> "EllBase":
        """Create an ellipsoid from a scaling factor.

        Args:
            alpha: The scaling factor (kappa) for the unit shape matrix.
            x_center: The initial center point of the ellipsoid.

        Returns:
            A new ellipsoid initialized from a scalar scaling factor.
        """
        return cls(alpha, x_center)

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

    def _update_core(self, cut: Cut, cut_strategy) -> CutStatus:
        """Apply the cutting plane using the strategy-specific matrix update.

        Template Method: the three public update_* entry points delegate here;
        each concrete subclass implements the actual ellipsoid update.

        Args:
            cut: Tuple (gradient, beta) for the cut
            cut_strategy: Strategy function to compute (rho, sigma, delta)

        Returns:
            CutStatus indicating success or failure of the update

        Raises:
            NotImplementedError: If the concrete subclass does not override
                this method.
        """
        raise NotImplementedError
