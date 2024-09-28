"""
EllCalc (Ellipsoid Calculator)

The EllCalc class is a tool designed to perform calculations related to ellipsoids, which are mathematical shapes similar to stretched spheres. This code is part of an algorithm used in optimization problems, specifically for a method called the ellipsoid method.

The main purpose of this code is to provide various functions that calculate how to adjust or "cut" an ellipsoid based on certain input parameters. These calculations are used to gradually refine the search space in optimization problems, helping to find optimal solutions more efficiently.

The primary input for this class is the dimension of the problem space, represented by the parameter 'n' in the constructor. Other inputs vary depending on the specific calculation method being used, but generally include values like 'beta' (which represents a cut point) and 'tsq' (which is related to the tolerance or precision of the cut).

The outputs of the various calculation methods are typically tuples containing a status (indicating whether the calculation was successful, had no solution, or had no effect) and, if successful, a set of three float values. These values (often named rho, sigma, and delta) represent parameters used to update the ellipsoid in the optimization algorithm.

The class achieves its purpose through a series of mathematical calculations. It uses concepts from linear algebra and geometry to determine how to shrink or reshape the ellipsoid based on the input parameters. The exact calculations are quite complex, but they essentially determine where to make a "cut" in the ellipsoid and how to reshape it accordingly.

Some important logic flows in the code include:

1. Checking if the input parameters are valid and returning appropriate status codes if they're not.
2. Deciding between different types of cuts (single, parallel, central) based on the input.
3. Performing specific calculations for each type of cut.

The code also includes a helper class (EllCalcCore) that likely handles some of the more complex mathematical operations.

Overall, this code serves as a crucial component in an optimization algorithm, providing the mathematical backbone for adjusting the search space as the algorithm progresses towards finding an optimal solution. While the underlying math is complex, the code encapsulates this complexity into methods that can be easily used by other parts of the optimization algorithm.
"""

from math import sqrt
from typing import Optional, Tuple

from .ell_calc_core import EllCalcCore
from .ell_config import CutStatus


class EllCalc:
    """The `EllCalc` class is used for calculating ellipsoid parameters and has attributes
    for storing constants and configuration options.

    Examples:
        >>> from ellalgo.ell_calc import EllCalc
        >>> calc = EllCalc(3)
    """

    use_parallel_cut: bool = True
    _n_f: float
    helper: EllCalcCore

    def __init__(self, n: int) -> None:
        """
        The function initializes several variables based on the input value.

        :param n: The parameter `n` represents an integer value. It is used to initialize the `EllCalc` object
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
        """single deep cut or parallel cut

        The `calc_single_or_parallel` function calculates either a single deep cut or a parallel cut based on
        the input parameters.

        :param beta: The parameter `beta` can be of type `int`, `float`, or a list of two elements

        :param tsq: The `tsq` parameter is a floating-point number that represents the square of the
            tolerance for the ellipsoid algorithm. It is used in the calculations performed by the
            `calc_single_or_parallel` method

        :type tsq: float

        :return: The function `calc_single_or_parallel` returns a tuple containing the following elements:

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
        """single central cut or parallel cut

        The function `calc_single_or_parallel_central_cut` calculates either a single central cut or a parallel cut
        based on the input parameters.

        :param beta: The parameter `beta` is of type `_type_` and represents some value. The specific
            details of its purpose and usage are not provided in the code snippet

        :param tsq: tsq is a float value representing the squared tau-value

        :type tsq: float

        :return: a tuple containing the following elements:
            1. CutStatus: The status of the cut calculation.
            2. float: The calculated value.
            3. float: The calculated value.
            4. float: The calculated value.

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(4)
            >>> calc.calc_single_or_parallel_central_cut([0, 0.11], 0.01)
            (<CutStatus.Success: 0>, (0.01897790039191521, 0.3450527343984584, 1.0549907942519101))
        """
        if isinstance(beta, (int, float)) or len(beta) < 2 or not self.use_parallel_cut:
            return (CutStatus.Success, self.helper.calc_central_cut(sqrt(tsq)))
        return (CutStatus.Success, self.helper.calc_parallel_central_cut(beta[1], tsq))

    def calc_parallel(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """parallel deep cut

        The function `calc_parallel` calculates the parallel deep cut based on the given parameters.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float representing the value of tsq
        :type tsq: float
        :return: The function `calc_parallel` returns a tuple of type `Tuple[CutStatus, Optional[Tuple[float, float, float]]]`.
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
        """Deep Cut

        The function calculates the deep cut based on the given beta and tsq values.

        :param beta: The parameter `beta` represents a float value
        :type beta: float
        :param tsq: tsq is the square of the value of tau
        :type tsq: float
        :return: The function `calc_bias_cut` returns a tuple of four values: `CutStatus`, `float`, `float`, `float`.

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
        """single deep cut or parallel cut (discrete)

        The function `calc_single_or_parallel_q` calculates the deep cut or parallel cut based on the input
        parameters `beta` and `tsq`.

        :param beta: The parameter `beta` can be either a single value (int or float) or a list of two values
        :param tsq: tsq is a float value representing the square of the threshold value
        :type tsq: float
        :return: The function `calc_single_or_parallel_q` returns a tuple containing four elements: `CutStatus`, `float`, `float`, and `float`.
        """
        if isinstance(beta, (int, float)):
            return self.calc_bias_cut_q(beta, tsq)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_bias_cut_q(beta[0], tsq)
        return self.calc_parallel_q(beta[0], beta[1], tsq)

    def calc_parallel_q(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[CutStatus, Optional[Tuple[float, float, float]]]:
        """Parallel deep cut (discrete)

        The function `calc_parallel_q` calculates the parallel deep cut for a given set of parameters.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float value that represents the square of a variable
        :type tsq: float
        :return: The function `calc_parallel_q` returns a tuple of type `Tuple[CutStatus, float, float, float]`.
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
        """Deep Cut (discrete)

        The function `calc_bias_cut_q` calculates the deep cut for a given beta and tsq value.

        :param beta: The parameter `beta` represents a float value
        :type beta: float
        :param tsq: tsq is the square of the threshold value. It is a float value that represents the
            threshold squared
        :type tsq: float
        :return: The function `calc_bias_cut_q` returns a tuple of four values: `CutStatus`, `float`, `float`, `float`.

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
