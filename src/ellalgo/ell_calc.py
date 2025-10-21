"""
EllCalc (Ellipsoid Calculator)

The EllCalc class is a tool designed to perform calculations related to ellipsoids,
which are mathematical shapes similar to stretched spheres. This code is part of
an algorithm used in optimization problems, specifically for a method called the
ellipsoid method.

The main purpose of this code is to provide various functions that calculate how
to adjust or "cut" an ellipsoid based on certain input parameters. These
calculations are used to gradually refine the search space in optimization
problems, helping to find optimal solutions more efficiently.

The primary input for this class is the dimension of the problem space,
represented by the parameter 'n' in the constructor. Other inputs vary
depending on the specific calculation method being used, but generally include
values like 'beta' (which represents a cut point) and 'tsq' (which is related
to the tolerance or precision of the cut).

The outputs of the various calculation methods are typically tuples containing a
status (indicating whether the calculation was successful, had no solution, or
had no effect) and, if successful, a set of three float values. These values
(often named rho, sigma, and delta) represent parameters used to update the
ellipsoid in the optimization algorithm.

The class achieves its purpose through a series of mathematical calculations. It
uses concepts from linear algebra and geometry to determine how to shrink or
reshape the ellipsoid based on the input parameters. The exact calculations are
quite complex, but they essentially determine where to make a "cut" in the
ellipsoid and how to reshape it accordingly.

Some important logic flows in the code include:

1. Checking if the input parameters are valid and returning appropriate status codes if they're not.
2. Deciding between different types of cuts (single, parallel, central) based on the input.
3. Performing specific calculations for each type of cut.

The code also includes a helper class (EllCalcCore) that likely handles some of the more complex mathematical operations.

Overall, this code serves as a crucial component in an optimization algorithm,
providing the mathematical backbone for adjusting the search space as the
algorithm progresses towards finding an optimal solution. While the underlying
math is complex, the code encapsulates this complexity into methods that can be
easily used by other parts of the optimization algorithm.
"""

from math import sqrt
from typing import Optional, Tuple

from .ell_calc_core import EllCalcCore
from .ell_config import CutStatus


class EllCalc:
    """Ellipsoid Calculator for the Ellipsoid Method.

    The `EllCalc` class is a core component of the ellipsoid method, responsible for
    calculating the parameters required to update the search ellipsoid at each iteration.
    It provides methods for various types of cutting planes, including deep cuts,
    parallel cuts, and central cuts.

    - A **deep cut** is a hyperplane that cuts the ellipsoid, excluding a portion of
      the search space that does not contain the optimal solution.
    - A **parallel cut** uses two parallel hyperplanes to cut the ellipsoid, which can
      be more efficient in certain situations.
    - A **central cut** is a special case of a deep cut that passes through the center
      of the ellipsoid.

    The class uses an instance of `EllCalcCore` to perform the low-level mathematical
    computations, while it handles the higher-level logic for selecting the appropriate
    cut type and returning the results.
    """

    use_parallel_cut: bool = True  # Flag to enable/disable parallel cut optimization
    _n_f: float  # Dimension of the space as a float
    helper: EllCalcCore  # Helper class for core calculations

    def __init__(self, n: int) -> None:
        """
        Initialize the EllCalc instance with the given dimension.

        The constructor sets up the necessary parameters for ellipsoid calculations,
        including storing the dimension and initializing the helper class.

        Args:            n (int): The dimension of the problem space (must be >= 2).

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(3)
            >>> calc._n_f
            3.0
        """
        assert n >= 2  # do not accept one-dimensional
        self._n_f = float(n)
        self.helper = EllCalcCore(n)

    def calc_single_or_parallel(
        self, beta, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """
        Calculates the parameters for either a single deep cut or a parallel cut.

        This method acts as a dispatcher, determining whether to perform a single
        deep cut or a parallel cut based on the `beta` parameter.

        - If `beta` is a single float, a single deep cut is performed.
        - If `beta` is a list or tuple with two elements, a parallel cut is performed
          using `beta[0]` and `beta[1]` as the two cut parameters.
        - The `use_parallel_cut` flag can be used to disable parallel cuts, in which
          case a single deep cut will be used as a fallback.

        Args:
            beta: The cut parameter(s). Can be a single float for a deep cut, or
                a list/tuple of two floats for a parallel cut.
            tsq: The square of the tolerance (τ²), used in the cut calculations.

        Returns:
            A tuple containing the `CutStatus` and an optional tuple of the
            update parameters (rho, sigma, delta) if the cut is successful.
        """
        if isinstance(beta, (int, float)):
            return self.calc_bias_cut(beta, tsq)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_bias_cut(beta[0], tsq)
        return self.calc_parallel(beta[0], beta[1], tsq)

    def calc_single_or_parallel_central_cut(
        self, beta, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Calculate either a single central cut or a parallel central cut.

        Similar to calc_single_or_parallel but specifically for central cuts (cuts passing through
        the center of the ellipsoid).

        Args:
            beta: Either a single numeric value or a list of two values.
            tsq: The square of the tolerance parameter (τ²).

        Returns:
            A tuple containing status and optional result values.

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> from ellalgo.ell_config import CutStatus
            >>> from pytest import approx
            >>> calc = EllCalc(4)
            >>> status, result = calc.calc_single_or_parallel_central_cut([0, 0.11], 0.01)
            >>> status
            <CutStatus.Success: 0>
            >>> result[0] == approx(0.02)
            True
        """
        if isinstance(beta, (int, float)) or len(beta) < 2 or not self.use_parallel_cut:
            return (CutStatus.Success, self.helper.calc_central_cut(sqrt(tsq)))
        if beta[1] < 0.0:
            return (CutStatus.NoSoln, None)
        b1sq = beta[1] * beta[1]
        if tsq <= b1sq:
            return (CutStatus.Success, self.helper.calc_central_cut(sqrt(tsq)))
        return (CutStatus.Success, self.helper.calc_parallel_central_cut(beta[1], tsq))

    def calc_parallel(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Calculate parameters for a parallel deep cut.

        A parallel cut involves two parallel hyperplanes cutting the ellipsoid. This method
        calculates the transformation parameters for such a cut after validating the inputs.

        Args:
            beta0 (float): First cut parameter (lower bound).
            beta1 (float): Second cut parameter (upper bound).
            tsq (float): Square of the tolerance parameter (τ²).

        Returns:
            Status and optional result tuple.

        The method first checks if beta1 < beta0 (invalid case), then checks if the cut
        would be outside the ellipsoid (tsq ≤ b1sq), and falls back to a single cut if so.
        Otherwise, it calculates the parallel cut parameters.
        """
        if beta1 < beta0:
            return (CutStatus.NoSoln, None)  # no sol'n
        b1sq = beta1 * beta1
        if beta1 > 0.0 and tsq <= b1sq:
            return self.calc_bias_cut(beta0, tsq)
        return (
            CutStatus.Success,
            self.helper.calc_parallel_bias_cut(beta0, beta1, tsq),
        )

    def calc_bias_cut(
        self, beta: float, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Calculate parameters for a single deep cut.

        A deep cut is a hyperplane cut that doesn't necessarily pass through the center
        of the ellipsoid. This method validates the input and calculates the transformation
        parameters if valid.

        Args:
            beta (float): Cut parameter (must be >= 0).
            tsq (float): Square of the tolerance parameter (τ²).

        Returns:
            Status and optional result tuple.

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> from ellalgo.ell_config import CutStatus
            >>> calc = EllCalc(3)
            >>> status, result = calc.calc_bias_cut(1.0, 4.0)
            >>> status
            <CutStatus.Success: 0>
            >>> result[0]
            1.25
            >>> status, result = calc.calc_bias_cut(0.0, 4.0)
            >>> status
            <CutStatus.Success: 0>
            >>> result[0]
            0.5
            >>> status, result = calc.calc_bias_cut(1.5, 2.0)
            >>> status
            <CutStatus.NoSoln: 1>
        """
        assert beta >= 0.0
        bsq = beta * beta
        if tsq < bsq:
            return (CutStatus.NoSoln, None)  # no sol'n
        tau = sqrt(tsq)
        return (
            CutStatus.Success,
            self.helper.calc_bias_cut(beta, tau),
        )

    def calc_single_or_parallel_q(
        self, beta, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Calculate either single or parallel deep cut (discrete version).

        This is a variant of calc_single_or_parallel designed for discrete optimization
        problems, with additional checks for numerical stability.

        Args:
            beta: Either a single numeric value or a list of two values.
            tsq (float): Square of the tolerance parameter (τ²).

        Returns:
            Status and optional result tuple.
        """
        if isinstance(beta, (int, float)):
            return self.calc_bias_cut_q(beta, tsq)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_bias_cut_q(beta[0], tsq)
        return self.calc_parallel_q(beta[0], beta[1], tsq)

    def calc_parallel_q(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Calculate parallel deep cut (discrete optimization version).

        This version includes additional checks for numerical stability in discrete
        optimization problems, specifically checking if eta ≤ 0.0.

        Args:
            beta0 (float): First cut parameter.
            beta1 (float): Second cut parameter.
            tsq (float): Square of the tolerance parameter (τ²).

        Returns:
            Status and optional result tuple.
        """
        if beta1 < beta0:
            return (CutStatus.NoSoln, None)  # no sol'n
        b1sq = beta1 * beta1
        if beta1 > 0.0 and tsq <= b1sq:
            return self.calc_bias_cut_q(beta0, tsq)
        b0b1 = beta0 * beta1
        eta = tsq + self._n_f * b0b1
        if eta <= 0.0:  # for discrete optimization
            return (CutStatus.NoEffect, None)  # no effect
        return (
            CutStatus.Success,
            self.helper.calc_parallel_bias_cut_fast(beta0, beta1, tsq, b0b1, eta),
        )

    def calc_bias_cut_q(
        self, beta: float, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Calculate deep cut (discrete optimization version).

        This version includes additional checks for numerical stability in discrete
        optimization problems, specifically checking if eta ≤ 0.0.

        Args:
            beta (float): Cut parameter.
            tsq (float): Square of the tolerance parameter (τ²).

        Returns:
            Status and optional result tuple.

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> from ellalgo.ell_config import CutStatus
            >>> calc = EllCalc(3)
            >>> status, result = calc.calc_bias_cut_q(0.0, 4.0)
            >>> status
            <CutStatus.Success: 0>
            >>> result[0]
            0.5
            >>> status, result = calc.calc_bias_cut_q(1.5, 2.0)
            >>> status
            <CutStatus.NoSoln: 1>
            >>> status, result = calc.calc_bias_cut_q(-1.5, 4.0)
            >>> status
            <CutStatus.NoEffect: 2>
        """
        tau = sqrt(tsq)
        if tau < beta:
            return (CutStatus.NoSoln, None)  # no sol'n
        eta = tau + self._n_f * beta
        if eta <= 0.0:
            return (CutStatus.NoEffect, None)
        return (
            CutStatus.Success,
            self.helper.calc_bias_cut_fast(beta, tau, eta),
        )
