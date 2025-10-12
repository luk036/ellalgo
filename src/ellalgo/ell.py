"""
Ell Class

This code defines a class called Ell which represents an ellipsoidal search space.
The purpose of this class is to provide methods for updating and manipulating an
ellipsoid, which is a mathematical shape used in certain optimization algorithms.

    The Ell class takes two main inputs when initialized:
    a value (which can be a number or a list of numbers) and an array xc. These inputs define the initial
shape and position of the ellipsoid. The class doesn't produce a specific output
on its own, but rather provides methods that can be used to modify and query the
ellipsoid's state.

The class achieves its purpose by maintaining several internal attributes that
represent the ellipsoid's properties, such as its center (_xc), a matrix (_mq),
and scaling factors (_kappa and _tsq). It then provides methods to update these
properties based on different types of "cuts" to the ellipsoid.

The main functionality of the Ell class revolves around three update methods:
update_bias_cut, update_central_cut, and update_q. These methods take a "cut"
as input, which is essentially a direction and a value that determine how to
modify the ellipsoid. The cuts are used to shrink or reshape the ellipsoid,
which is a key operation in certain optimization algorithms.

The core logic of these update methods is implemented in the private _update_core method. This method applies the cut to the ellipsoid by performing a series of mathematical operations. It calculates new values for the ellipsoid's center and shape based on the input cut and a specified cut strategy.

An important aspect of the code is its use of numpy, a library for numerical computations in Python. The class uses numpy arrays and matrix operations to efficiently perform the necessary calculations.

The class also includes some helper methods like xc() and tsq() that allow access to certain properties of the ellipsoid. These can be used to query the current state of the ellipsoid during an optimization process.

Overall, this code provides a flexible and efficient way to represent and
manipulate an ellipsoidal search space, which is a crucial component in certain
types of optimization algorithms. The class encapsulates the complex mathematics
involved in these operations, providing a clean interface for users of the class
to work with ellipsoids in their algorithms.
"""

from typing import Callable, Generic, Tuple, Union

import numpy as np

from .ell_calc import EllCalc
from .ell_config import CutStatus
from .ell_typing import ArrayType

# Type aliases for better code readability
Mat = np.ndarray
CutChoice = Union[float, ArrayType]  # single or parallel cut
Cut = Tuple[ArrayType, CutChoice]  # A cut consists of a gradient and a beta value


# The `Ell` class represents an ellipsoidal search space.
class Ell(Generic[ArrayType]):
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

    def __init__(self, val, xc: ArrayType) -> None:
        """
        Initialize the ellipsoid with given parameters.

        The initialization can be done in two ways:
        1. With a scalar value (kappa) which creates a unit matrix
        2. With a list of values which creates a diagonal matrix

        Args:
            val: Either a scalar (kappa) or a list of values for diagonal matrix
            xc: The initial center point of the ellipsoid

        The method:
        1. Determines the dimension from xc
        2. Creates a helper object for calculations
        3. Sets the center point
        4. Initializes tsq to 0
        5. Sets either kappa with unit matrix or diagonal matrix based on val type
        """
        ndim = len(xc)
        self.helper = EllCalc(ndim)
        self._xc = xc
        self._tsq = 0.0
        if isinstance(val, (int, float)):
            # Case 1: val is a scalar (kappa), create identity matrix
            self._kappa = val
            self._mq = np.eye(ndim)
        else:
            # Case 2: val is a list/array, create diagonal matrix
            self._kappa = 1.0
            self._mq = np.diag(val)

    def xc(self) -> np.ndarray:
        """
        Getter method for the ellipsoid's center point.

        Returns:
            The current center point (_xc) of the ellipsoid
        """
        return self._xc

    def set_xc(self, xc: ArrayType) -> None:
        """
        Setter method for the ellipsoid's center point.

        Args:
            xc: The new center point for the ellipsoid
        """
        self._xc = xc

    def tsq(self) -> float:
        """
        Getter method for the tsq value.

        tsq represents the measure of distance between current center (xc) and optimal point (x*).
        It's calculated as kappa * omega, where omega is grad^T * M * grad.

        Returns:
            The current tsq value
        """
        return self._tsq

    def update_bias_cut(self, cut) -> CutStatus:
        """
        Update the ellipsoid using a bias cut (deep cut) strategy.

        A bias cut is a general cut that can be either deep or shallow.
        This method delegates to _update_core with the standard cut strategy.

        Args:
            cut: A tuple containing (gradient, beta) for the cut

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell.update_bias_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel)

    def update_central_cut(self, cut) -> CutStatus:
        """
        Update the ellipsoid using a central cut strategy.

        A central cut is a special case where beta = 0, meaning the cut passes
        exactly through the center of the current ellipsoid.

        Args:
            cut: A tuple containing (gradient, beta) for the cut

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 0.0)
            >>> status = ell.update_central_cut(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_central_cut)

    def update_q(self, cut) -> CutStatus:
        """
        Update the ellipsoid using a non-central cut strategy for Q.

        This is used for non-central cuts (either deep or shallow) in Q space.

        Args:
            cut: A tuple containing (gradient, beta) for the cut

        Returns:
            CutStatus indicating success or failure of the update

        Examples:
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), -0.01)
            >>> status = ell.update_q(cut)
            >>> print(status)
            CutStatus.Success
        """
        return self._update_core(cut, self.helper.calc_single_or_parallel_q)

    # private:

    def _update_core(self, cut, cut_strategy: Callable) -> CutStatus:
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
            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel)
            >>> print(status)
            CutStatus.Success

            >>> ell = Ell(1.0, [1.0, 1.0, 1.0, 1.0])
            >>> cut = (np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
            >>> status = ell._update_core(cut, ell.helper.calc_single_or_parallel_central_cut)
            >>> print(status)
            CutStatus.Success
        """
        grad, beta = cut
        if np.all(grad == 0.0):
            raise ValueError("Gradient cannot be a zero vector.")
        # Calculate M * grad (matrix-vector multiplication)
        grad_t = self._mq @ grad  # n^2 multiplications
        # Calculate grad^T * (M * grad)
        omega = grad.dot(grad_t)  # n multiplications
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
