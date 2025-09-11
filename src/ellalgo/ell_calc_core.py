"""This module contains the `EllCalcCore` class.

This code defines a class called EllCalcCore, which is designed to perform calculations related to ellipsoids in mathematics. An ellipsoid is a 3D shape that's like a stretched or squashed sphere. The class provides methods to calculate various parameters of ellipsoids under different conditions.

The main purpose of this code is to provide a set of tools for working with ellipsoids in optimization problems. It takes in various numerical inputs representing different aspects of an ellipsoid or cuts through it, and produces output values that describe how the ellipsoid should be adjusted or analyzed.

The class is initialized with a single input 'n_f', which represents the dimension of the space the ellipsoid exists in. This value is used to set up several constant values that are used in later calculations.

The class provides several methods, each performing a different type of calculation:

1. calc_central_cut: This calculates parameters for a cut through the center of the ellipsoid.
2. calc_bias_cut and calc_bias_cut_fast: These calculate parameters for a cut that doesn't go through the center.
3. calc_parallel_central_cut and calc_parallel_central_cut_old: These handle cuts that are parallel to a central cut.
4. calc_parallel_bias_cut, calc_parallel_bias_cut_fast, and calc_parallel_bias_cut_old: These deal with parallel cuts that don't go through the center.

Each of these methods takes in one or more numerical inputs (like 'tau', 'beta', 'tsq') that represent different aspects of the cut or the ellipsoid. They then perform a series of mathematical calculations using these inputs and the constants set up during initialization. The calculations involve basic arithmetic, square roots, and some more complex formulas specific to ellipsoid geometry.

The output of each method is typically a tuple of three float values, often represented as (rho, sigma, delta). These values describe how the ellipsoid should be adjusted based on the cut that was calculated.

The code achieves its purpose by encapsulating all these complex mathematical calculations into easy-to-use methods. A programmer can create an instance of EllCalcCore and then call these methods as needed, without having to understand all the underlying mathematics.

The main logic flow in this code is from the input parameters, through the mathematical calculations, to the output tuple. The data transformations happening are primarily mathematical: converting the input parameters into the desired output parameters using the formulas of ellipsoid geometry.

This code is a tool for more complex algorithms that might be using these ellipsoid calculations as part of a larger optimization or analysis process. It provides a clean, object-oriented way to perform these specific mathematical operations.
"""

from math import sqrt
from typing import Tuple


class EllCalcCore:
    """The `EllCalcCore` class is used for calculating ellipsoid parameters.

    This class provides methods to compute various parameters (rho, sigma, delta) that define
    how an ellipsoid should be transformed when cut by different types of hyperplanes. These
    parameters are essential in ellipsoid-based optimization algorithms.

    Examples:
        >>> calc = EllCalcCore(3)
    """

    _n_f: float
    _half_n: float
    _n_plus_1: float
    _cst0: float
    _cst1: float
    _cst2: float
    _cst3: float

    def __init__(self, n_f: float) -> None:
        """Initialize EllCalcCore instance.

        The __init__ method initializes the EllCalcCore object with the provided
        n_f parameter. This sets up internal variables used in calculations.

        The initialization computes several constants that are frequently used in
        the ellipsoid calculations to avoid repeated computation:
        - _half_n: Half of the dimension n_f
        - _n_plus_1: n_f + 1
        - _inv_n: 1/n_f
        - _n_sq: n_f squared
        - _cst0 to _cst3: Various precomputed constants used in cut calculations

        :param n_f: Float value representing the dimension of the space
        :type n_f: float

        Examples:
            >>> calc = EllCalcCore(3)
            >>> calc._n_f
            3
            >>> calc._half_n
            1.5
            >>> calc._cst0
            0.25
            >>> calc._cst1
            1.125
            >>> calc._cst2
            0.5
            >>> calc._cst3
            0.75
        """
        self._n_f = n_f
        self._half_n = self._n_f / 2.0
        self._n_plus_1 = self._n_f + 1.0
        self._inv_n = 1.0 / self._n_f
        self._n_sq = self._n_f * self._n_f
        self._cst0 = 1.0 / self._n_plus_1
        self._cst1 = self._n_sq / (self._n_sq - 1.0)
        self._cst2 = 2.0 * self._cst0
        self._cst3 = self._n_f * self._cst0

    def calc_central_cut(self, tau: float) -> Tuple[float, float, float]:
        r"""Calculate the central cut values.

        The `calc_central_cut` method calculates the central cut values ρ, σ, δ
        based on the input tau value. A central cut is a hyperplane that passes
        through the center of the ellipsoid.

        The parameters returned are:
        - ρ (rho): Determines the shift of the ellipsoid center
        - σ (sigma): Determines how much to scale the ellipsoid
        - δ (delta): Determines how much to stretch the ellipsoid

        :param tau: The distance parameter for the central cut
        :type tau: float
        :return: Tuple of (ρ, σ, δ) values for the central cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'      |      `.
                 /        |        \
                .         |         .
                |                   |
                |         .         |
                |                   |
                :\        |        /:
                | `._     |     _.' |
                |    '-.......-'    |
                |         |         |
               "-τ"       0        +τ

                      2
                σ = ─────
                    n + 1

                      τ
                ϱ = ─────
                    n + 1

                       2
                      n
                δ = ──────
                     2
                    n  - 1

        Examples:
            >>> calc = EllCalcCore(3)
            >>> calc.calc_central_cut(4.0)
            (1.0, 0.5, 1.125)
        """
        rho = self._cst0 * tau
        sigma = self._cst2
        delta = self._cst1
        return (rho, sigma, delta)

    def calc_bias_cut_fast(
        self, beta: float, tau: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Calculates the deep cut ellipsoid parameters using precomputed eta values.

        This is an optimized version of calc_bias_cut that takes a precomputed eta value
        (η = τ + n⋅β) as input to avoid redundant calculations when eta is already known.

        The method computes parameters for a biased (non-central) cut of the ellipsoid,
        where the cut doesn't pass through the center. The parameters determine how to
        transform the ellipsoid after the cut.

        :param beta: The bias parameter (distance from center to cut hyperplane)
        :type beta: float
        :param tau: The distance parameter for the cut
        :type tau: float
        :param eta: Precomputed value of τ + n⋅β
        :type eta: float
        :return: Tuple of (ρ, σ, δ) values for the biased cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'    |        `.
                 /      |          \
                .       |           .
                |       |           |
                |       | .         |
                |       |           |
                :\      |          /:
                | `._   |       _.' |
                |    '-.......-'    |
                |       |           |
               "-τ"     "-β"       +τ

                      η
                ϱ = ─────
                    n + 1

                    2 ⋅ ϱ
                σ = ─────
                    τ + β

                       2       2    2
                      n       τ  - β
                δ = ────── ⋅  ───────
                     2           2
                    n  - 1      τ

        Examples:
            >>> calc = EllCalcCore(3)
            >>> calc.calc_bias_cut_fast(1.0, 2.0, 5.0)
            (1.25, 0.8333333333333334, 0.84375)
            >>> calc.calc_bias_cut_fast(0.0, 2.0, 2.0)
            (0.5, 0.5, 1.125)
        """
        alpha = beta / tau
        rho = self._cst0 * eta
        sigma = self._cst2 * eta / (tau + beta)
        delta = self._cst1 * (1.0 - alpha) * (1.0 + alpha)
        return (rho, sigma, delta)

    def calc_bias_cut(self, beta: float, tau: float) -> Tuple[float, float, float]:
        r"""Calculate deep cut values.

        Calculates the deep cut values ρ, σ, δ for given β and τ. This is the standard
        version that computes η internally before calling calc_bias_cut_fast.

        A deep cut is a generalization of a central cut where the cutting hyperplane
        doesn't necessarily pass through the center of the ellipsoid. The parameters
        determine how to transform the ellipsoid after the cut.

        :param beta: The bias parameter (distance from center to cut hyperplane)
        :type beta: float
        :param tau: The distance parameter for the cut
        :type tau: float
        :return: Tuple of (ρ, σ, δ) values for the biased cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'   |         `.
                 /     |           \
                .      |            .
                |      |            |
                |      |  .         |
                |      |            |
                :\     |           /:
                | `._  |        _.' |
                |    '-.......-'    |
                |      |            |
               "-τ"     "-β"       +τ

                η = τ + n ⋅ β

                      η
                ϱ = ─────
                    n + 1

                    2 ⋅ ϱ
                σ = ─────
                    τ + β

                       2       2    2
                      n       τ  - β
                δ = ────── ⋅  ───────
                     2           2
                    n  - 1      τ

        Examples:
            >>> calc = EllCalcCore(3)
            >>> calc.calc_bias_cut(1.0, 2.0)
            (1.25, 0.8333333333333334, 0.84375)
            >>> calc.calc_bias_cut(0.0, 2.0)
            (0.5, 0.5, 1.125)
        """
        return self.calc_bias_cut_fast(beta, tau, tau + self._n_f * beta)

    def calc_parallel_central_cut(
        self, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculate Parallel Central Cut (7 mul/div + 1 sqrt)

        Computes parameters for a cut that is parallel to a central cut but offset by beta1.
        This optimized version uses fewer operations than the old version.

        The method calculates:
        - ρ: Determines the shift of the ellipsoid center
        - σ: Scaling factor for the ellipsoid
        - δ: Stretching factor for the ellipsoid

        :param beta1: Offset parameter for the parallel cut
        :type beta1: float
        :param tsq: Square of the distance parameter (τ²)
        :type tsq: float
        :return: Tuple of (ρ, σ, δ) values for the parallel central cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'      |      `.
                 /  |     |        \
                .   |     |         .
                |   |               |
                |   |     .         |
                |   |               |
                :\  |     |        /:
                | `._     |     _.' |
                |   |'-.......-'    |
                |   |     |         |
               "-τ" "-β"  0        +τ
                      1

             2    2    2
            α  = β  / τ

                n    2
            k = ─ ⋅ α
                2
                       ___________
                      ╱ 2        2
            r = k + ╲╱ k  + 1 - α

                  β
            ϱ = ─────
                r + 1

                  2
            σ = ─────
                r + 1

                    r
            δ = ─────────
                r - 1 / n

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_central_cut(0.09, 0.01)
            (0.020941836487980856, 0.46537414417735234, 1.082031295477563)
        """
        a1sq = beta1 * beta1 / tsq
        k = self._half_n * a1sq
        r = k + sqrt(1.0 - a1sq + k * k)
        rho = beta1 / (r + 1.0)
        sigma = 2.0 / (r + 1.0)
        delta = r / (r - self._inv_n)
        return (rho, sigma, delta)

    def calc_parallel_central_cut_old(
        self, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculate Parallel Central Cut

        Original version of the parallel central cut calculation. This version
        uses more operations than the optimized version.

        The method calculates parameters for transforming an ellipsoid after
        a parallel central cut, where the cut is parallel to a central cut
        but offset by beta1.

        :param beta1: Offset parameter for the parallel cut
        :type beta1: float
        :param tsq: Square of the distance parameter (τ²)
        :type tsq: float
        :return: Tuple of (ρ, σ, δ) values for the parallel central cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'      |      `.
                 /  |     |        \
                .   |     |         .
                |   |               |
                |   |     .         |
                |   |               |
                :\  |     |        /:
                | `._     |     _.' |
                |   |'-.......-'    |
                |   |     |         |
               "-τ" "-β"  0        +τ
                      1
                           __________________________
                          ╱                         2
                         ╱                  ⎛     2⎞
                        ╱                   ⎜n ⋅ β ⎟
                       ╱   ⎛ 2    2⎞    2   ⎜     1⎟
                ξ =   ╱    ⎜τ  - β ⎟ ⋅ τ  + ⎜──────⎟
                    ╲╱     ⎝      1⎠        ⎝   2  ⎠

                                ⎛ 2    ⎞
                      n     2 ⋅ ⎝τ  - ξ⎠
                σ = ───── + ────────────
                    n + 1              2
                            (n + 1) ⋅ β
                                       1

                    σ ⋅ β
                         1
                ϱ = ──────
                       2

                         ⎛      2    ⎞
                         ⎜     β     ⎟
                     2   ⎜ 2    1   ξ⎟
                    n  ⋅ ⎜τ  - ── + ─⎟
                         ⎝      2   n⎠
                δ = ──────────────────
                       ⎛ 2    ⎞    2
                       ⎝n  - 1⎠ ⋅ τ

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_central_cut_old(0.09, 0.01)
            (0.02094183648798086, 0.46537414417735246, 1.082031295477563)
        """
        b1sq = beta1 * beta1
        # if tsq < b1sq or not self.use_parallel_cut:
        #     return self.calc_cc(tsq)
        # Core calculation
        a1sq = b1sq / tsq
        xi = sqrt(1.0 - a1sq + (self._half_n * a1sq) ** 2)
        sigma = self._cst3 + self._cst2 * (1.0 - xi) / a1sq
        rho = sigma * beta1 / 2.0
        delta = self._cst1 * (1.0 - a1sq / 2.0 + xi / self._n_f)
        return (rho, sigma, delta)

    def calc_parallel_bias_cut(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculation Parallel Deep Cut (15 mul/div + 1 sqrt)

        Computes parameters for a biased cut that is parallel to another biased cut.
        This version computes intermediate values (b0b1 and eta) internally before
        calling the fast version.

        The method calculates:
        - ρ: Center shift parameter
        - σ: Scaling parameter
        - δ: Stretching parameter

        :param beta0: First bias parameter (for the reference cut)
        :type beta0: float
        :param beta1: Second bias parameter (for the parallel cut)
        :type beta1: float
        :param tsq: Square of the distance parameter (τ²)
        :type tsq: float
        :return: Tuple of (ρ, σ, δ) values for the parallel biased cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'     |       `.
                 /  |    |         \
                .   |    |          .
                |   |    |          |
                |   |    |.         |
                |   |    |          |
                :\  |    |         /:
                | `._    |      _.' |
                |   |'-.......-'    |
                |   |    |          |
               "-τ" "-β" "-β"      +τ
                      1    0
                 2
            η = τ  + n ⋅ β  ⋅ β
                          0    1
                β  + β
                 0    1
            β = ───────
                   2

                1   ⎛ 2          ⎞        2
            h = ─ ⋅ ⎜τ  + β  ⋅ β ⎟ + n ⋅ β
                2   ⎝      0    1⎠
                       _____________________
                      ╱ 2                  2
            k = h + ╲╱ h  - (n + 1) ⋅ η ⋅ β

                  1     η
            σ = ───── = ─
                μ + 1   k

            1     η
            ─ = ─────
            μ   k - η

            ϱ = β ⋅ σ

                 2    2   1   ⎛ 2              ⎞
            δ ⋅ τ  = τ  + ─ ⋅ ⎜β  ⋅ σ - β  ⋅ β ⎟
                          μ   ⎝          0    1⎠

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_bias_cut(0.01, 0.11, 0.01)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_bias_cut(-0.25, 0.25, 1.0)
            (0.0, 0.8, 1.25)
            >>> calc.calc_parallel_bias_cut(0.0, 0.09, 0.01)
            (0.020941836487980856, 0.46537414417735234, 1.082031295477563)
        """
        b0b1 = beta0 * beta1
        return self.calc_parallel_bias_cut_fast(
            beta0, beta1, tsq, b0b1, tsq + self._n_f * b0b1
        )

    def calc_parallel_bias_cut_fast(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Calculation Parallel Deep Cut (13 mul/div + 1 sqrt)

        Optimized version of parallel biased cut calculation that takes precomputed
        b0b1 (β₀β₁) and eta (η) values as input to reduce computation.

        This version is more efficient when the calling code can precompute these
        values, saving redundant calculations.

        :param beta0: First bias parameter (for the reference cut)
        :type beta0: float
        :param beta1: Second bias parameter (for the parallel cut)
        :type beta1: float
        :param tsq: Square of the distance parameter (τ²)
        :type tsq: float
        :param b0b1: Precomputed product of beta0 and beta1 (β₀β₁)
        :type b0b1: float
        :param eta: Precomputed value of τ² + n⋅β₀β₁
        :type eta: float
        :return: Tuple of (ρ, σ, δ) values for the parallel biased cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'     |       `.
                 /  |    |         \
                .   |    |          .
                |   |    |          |
                |   |    |.         |
                |   |    |          |
                :\  |    |         /:
                | `._    |      _.' |
                |   |'-.......-'    |
                |   |    |          |
               "-τ" "-β" "-β"      +τ
                      1    0

                β  + β
                 0    1
            β = ───────
                   2

                1   ⎛ 2          ⎞        2
            h = ─ ⋅ ⎜τ  + β  ⋅ β ⎟ + n ⋅ β
                2   ⎝      0    1⎠
                       _____________________
                      ╱ 2                  2
            k = h + ╲╱ h  - (n + 1) ⋅ η ⋅ β

                  1     η
            σ = ───── = ─
                μ + 1   k

            1     η
            ─ = ─────
            μ   k - η

            ϱ = β ⋅ σ

                 2    2   1   ⎛ 2              ⎞
            δ ⋅ τ  = τ  + ─ ⋅ ⎜β  ⋅ σ - β  ⋅ β ⎟
                          μ   ⎝          0    1⎠

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_bias_cut_fast(0.11, 0.01, 0.01, 0.0011, 0.0144)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_bias_cut_fast(-0.25, 0.25, 1.0, -0.0625, 0.75)
            (0.0, 0.8, 1.25)
        """
        bavg = 0.5 * (beta0 + beta1)
        bavgsq = bavg * bavg
        h = 0.5 * (tsq + b0b1) + self._n_f * bavgsq
        k = h + sqrt(h * h - self._n_plus_1 * eta * bavgsq)
        sigma = eta / k
        inv_mu = eta / (k - eta)
        rho = bavg * sigma
        delta = (tsq + inv_mu * (bavgsq * sigma - b0b1)) / tsq
        return (rho, sigma, delta)

    def calc_parallel_bias_cut_old(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculation Parallel Deep Cut

        Original version of the parallel biased cut calculation. This version
        uses a different mathematical formulation that requires more computations
        than the optimized versions.

        The method calculates parameters for transforming an ellipsoid after
        a parallel biased cut, where the cut is parallel to another biased cut.

        :param beta0: First bias parameter (for the reference cut)
        :type beta0: float
        :param beta1: Second bias parameter (for the parallel cut)
        :type beta1: float
        :param tsq: Square of the distance parameter (τ²)
        :type tsq: float
        :return: Tuple of (ρ, σ, δ) values for the parallel biased cut
        :rtype: Tuple[float, float, float]

        .. svgbob::
           :align: center

                    _.-'''''''-._
                  ,'     |       `.
                 /  |    |         \
                .   |    |          .
                |   |    |          |
                |   |    |.         |
                |   |    |          |
                :\  |    |         /:
                | `._    |      _.' |
                |   |'-.......-'    |
                |   |    |          |
               "-τ" "-β" "-β"      +τ
                      1    0

                      2    2
                ζ  = τ  - β
                 0         0

                      2    2
                ζ  = τ  - β
                 1         1
                           __________________________
                          ╱                         2
                         ╱           ⎛    ⎛ 2    2⎞⎞
                        ╱            ⎜n ⋅ ⎜β  - β ⎟⎟
                       ╱             ⎜    ⎝ 1    0⎠⎟
                ξ =   ╱    ζ  ⋅ ζ  + ⎜─────────────⎟
                    ╲╱      0    1   ⎝      2      ⎠

                                ⎛ 2              ⎞
                            2 ⋅ ⎜τ  + β  ⋅ β  - ξ⎟
                      n         ⎝      0    1    ⎠
                σ = ───── + ──────────────────────
                    n + 1                       2
                             (n + 1) ⋅ ⎛β  + β ⎞   <---- Oop!!!
                                       ⎝ 0    1⎠

                    σ ⋅ ⎛β  + β ⎞
                        ⎝ 0    1⎠
                ϱ = ─────────────
                          2

                         ⎛ζ  + ζ     ⎞
                     2   ⎜ 0    1   ξ⎟
                    n  ⋅ ⎜─────── + ─⎟
                         ⎝   2      n⎠
                δ = ──────────────────
                       ⎛ 2    ⎞    2
                       ⎝n  - 1⎠ ⋅ τ

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_bias_cut_old(0.11, 0.01, 0.01)
            (0.02722850906828212, 0.4538084844713687, 1.0443438549074862)
        """
        b0b1 = beta0 * beta1
        b0sq = beta0 * beta0
        b1sq = beta1 * beta1
        zeta0 = tsq - b0sq
        zeta1 = tsq - b1sq
        xi = sqrt(zeta0 * zeta1 + (self._half_n * (b1sq - b0sq)) ** 2)
        bsumsq = (beta0 + beta1) ** 2
        sigma = self._cst3 + self._cst2 * (tsq + b0b1 - xi) / bsumsq
        rho = sigma * (beta0 + beta1) / 2.0
        delta = self._cst1 * ((zeta0 + zeta1) / 2.0 + xi / self._n_f) / tsq
        return (rho, sigma, delta)
