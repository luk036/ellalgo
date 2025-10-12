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
    """The `EllCalc` class is used for calculating ellipsoid parameters and has attributes
    for storing constants and configuration options.

    This class serves as the main interface for performing ellipsoid calculations in optimization
    algorithms. It provides methods for different types of cuts (single, parallel, central) and
    handles the logic for selecting the appropriate cut type based on input parameters.

    The class uses an instance of EllCalcCore to perform the actual mathematical computations,
    while handling the higher-level logic and status reporting.

    Examples:
        >>> from ellalgo.ell_calc import EllCalc
        >>> calc = EllCalc(3)
        >>> calc._n_f
        3.0
    """

    use_parallel_cut: bool = True  # Flag to enable/disable parallel cut optimization
    _n_f: float  # Dimension of the space as a float
    helper: EllCalcCore  # Helper class for core calculations

    def __init__(self, n: int) -> None:
        """
        Initialize the EllCalc instance with the given dimension.

        The constructor sets up the necessary parameters for ellipsoid calculations,
        including storing the dimension and initializing the helper class.

        :param n: The dimension of the problem space (must be ≥ 2)
        :type n: int

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
        """Calculate either a single deep cut or a parallel cut based on input parameters.

        This method serves as a dispatcher that chooses between single cut and parallel cut
        calculations based on the type of beta parameter provided.

        :param beta: Either a single numeric value (for single cut) or a list of two values (for parallel cut)
        :param tsq: The square of the tolerance parameter (τ²) used in the cut calculations
        :type tsq: float
        :return: A tuple containing:
                 - CutStatus: indicating success or failure
                 - Optional tuple of (rho, sigma, delta) if successful

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(3)
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

        :param beta: Either a single numeric value or a list of two values
        :param tsq: The square of the tolerance parameter (τ²)
        :type tsq: float
        :return: A tuple containing status and optional result values

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(4)
            >>> calc.calc_single_or_parallel_central_cut([0, 0.11], 0.01)
            (<CutStatus.Success: 0>, (0.020000000000000004, 0.4, 1.0666666666666667))
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

        :param beta0: First cut parameter (lower bound)
        :type beta0: float
        :param beta1: Second cut parameter (upper bound)
        :type beta1: float
        :param tsq: Square of the tolerance parameter (τ²)
        :type tsq: float
        :return: Status and optional result tuple

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

        :param beta: Cut parameter (must be ≥ 0)
        :type beta: float
        :param tsq: Square of the tolerance parameter (τ²)
        :type tsq: float
        :return: Status and optional result tuple

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(3)
            >>> calc.calc_bias_cut(1.0, 4.0)
            (<CutStatus.Success: 0>, (1.25, 0.8333333333333334, 0.84375))
            >>> calc.calc_bias_cut(0.0, 4.0)
            (<CutStatus.Success: 0>, (0.5, 0.5, 1.125))
            >>> calc.calc_bias_cut(1.5, 2.0)
            (<CutStatus.NoSoln: 1>, None)
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

        :param beta: Either a single numeric value or a list of two values
        :param tsq: Square of the tolerance parameter (τ²)
        :type tsq: float
        :return: Status and optional result tuple
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

        :param beta0: First cut parameter
        :type beta0: float
        :param beta1: Second cut parameter
        :type beta1: float
        :param tsq: Square of the tolerance parameter (τ²)
        :type tsq: float
        :return: Status and optional result tuple
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

        :param beta: Cut parameter
        :type beta: float
        :param tsq: Square of the tolerance parameter (τ²)
        :type tsq: float
        :return: Status and optional result tuple

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(3)
            >>> calc.calc_bias_cut_q(0.0, 4.0)
            (<CutStatus.Success: 0>, (0.5, 0.5, 1.125))
            >>> calc.calc_bias_cut_q(1.5, 2.0)
            (<CutStatus.NoSoln: 1>, None)
            >>> calc.calc_bias_cut_q(-1.5, 4.0)
            (<CutStatus.NoEffect: 2>, None)
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


if __name__ == "__main__":
    from pytest import approx

    # Test cases for the parallel cut calculations
    ell_calc = EllCalc(4)
    status, _ = ell_calc.calc_parallel_q(0.07, 0.03, 0.01)
    assert status == CutStatus.NoSoln

    status, result = ell_calc.calc_parallel_q(0.0, 0.05, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)

    status, result = ell_calc.calc_parallel_q(0.05, 0.11, 0.01)
    assert status == CutStatus.Success
    assert result is not None
    rho, sigma, delta = result
    assert sigma == approx(0.8)
    assert rho == approx(0.06)
    assert delta == approx(0.8)
