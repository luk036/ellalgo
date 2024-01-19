"""This module contains the `EllCalcCore` class."""

from math import sqrt
from typing import Tuple


class EllCalcCore:
    """The `EllCalcCore` class is used for calculating ellipsoid parameters.

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

        :param n_f: Float value to initialize the instance with.
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
        based on the input tau value.

        :param tau: The tau value
        :type tau: float
        :return: Tuple of (ρ, σ, δ)
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

    def calc_deep_cut_fast(
        self, beta: float, tau: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Calculates the deep cut ellipsoid parameters using precomputed eta values.

        Given the beta, tau, and eta values, this method calculates the
        deep cut ellipsoid parameters rho, sigma, and delta using a precomputed eta value.
        that avoids explicitly calculating the intermediate eta value.
        This allows the deep cut to be computed faster.

        The rho, sigma, and delta values define the deep cut ellipsoid.

        :param beta: beta is a float representing the value of beta
        :type beta: float
        :param tau: tau is a float representing the value of tau
        :type tau: float
        :return: The function `calc_deep_cut` returns a tuple containing the following elements:

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
            >>> calc.calc_deep_cut_fast(1.0, 2.0, 5.0)
            (1.25, 0.8333333333333334, 0.84375)
            >>> calc.calc_deep_cut_fast(0.0, 2.0, 2.0)
            (0.5, 0.5, 1.125)
        """
        alpha = beta / tau
        rho = self._cst0 * eta
        sigma = self._cst2 * eta / (tau + beta)
        delta = self._cst1 * (1.0 - alpha) * (1.0 + alpha)
        return (rho, sigma, delta)

    def calc_deep_cut(self, beta: float, tau: float) -> Tuple[float, float, float]:
        r"""Calculate deep cut values.

        Calculates the deep cut values ρ, σ, δ for given β and τ.

        :param beta: beta is a float representing the value of beta
        :type beta: float
        :param tau: tau is a float representing the value of tau
        :type tau: float
        :return: The function `calc_deep_cut` returns a tuple containing the following elements:

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
            >>> calc.calc_deep_cut(1.0, 2.0)
            (1.25, 0.8333333333333334, 0.84375)
            >>> calc.calc_deep_cut(0.0, 2.0)
            (0.5, 0.5, 1.125)
        """
        return self.calc_deep_cut_fast(beta, tau, tau + self._n_f * beta)

    def calc_parallel_central_cut(
        self, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculate Parallel Central Cut (7 mul/div + 1 sqrt)

        The function `calc_parallel_central_cut` calculates the parallel central cut for given values of `beta1` and `tsq`.

        :param beta1: The parameter `beta1` represents a float value. It is used in the calculation of the central cut
        :type beta1: float
        :param tsq: The parameter `tsq` represents the square of a value
        :type tsq: float
        :return: The function `calc_parallel_central_cut` returns a tuple of four values: `CutStatus`, `float`, `float`, `float`.

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

        The function `calc_parallel_central_cut` calculates the parallel central cut for given values of `beta1` and `tsq`.

        :param beta1: The parameter `beta1` represents a float value. It is used in the calculation of the central cut
        :type beta1: float
        :param tsq: The parameter `tsq` represents the square of a value
        :type tsq: float
        :return: The function `calc_parallel_central_cut` returns a tuple of four values: `CutStatus`, `float`, `float`, `float`.

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

    def calc_parallel_deep_cut(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculation Parallel Deep Cut (15 mul/div + 1 sqrt)

        The `calc_parallel_deep_cut` function calculates various values based on the input parameters and returns
        them as a tuple.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float representing the square of the value tau
        :type tsq: float
        :return: a tuple with three elements.

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
            >>> calc.calc_parallel_deep_cut(0.01, 0.11, 0.01)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_deep_cut(-0.25, 0.25, 1.0)
            (0.0, 0.8, 1.25)
            >>> calc.calc_parallel_deep_cut(0.0, 0.09, 0.01)
            (0.020941836487980856, 0.46537414417735234, 1.082031295477563)
        """
        b0b1 = beta0 * beta1
        return self.calc_parallel_deep_cut_fast(
            beta0, beta1, tsq, b0b1, tsq + self._n_f * b0b1
        )

    def calc_parallel_deep_cut_fast(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Calculation Parallel Deep Cut (13 mul/div + 1 sqrt)

        The `calc_parallel_deep_cut_fast` function calculates various values based on the input parameters and returns them as a tuple.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float representing the square of the value tau
        :type tsq: float
        :param b0b1: The parameter `b0b1` represents a float value
        :type b0b1: float
        :param eta: The parameter `eta` represents a float value
        :type eta: float
        :return: a tuple with three elements.

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
            >>> calc.calc_parallel_deep_cut_fast(0.11, 0.01, 0.01, 0.0011, 0.0144)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_deep_cut_fast(-0.25, 0.25, 1.0, -0.0625, 0.75)
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

    def calc_parallel_deep_cut_old(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Calculation Parallel Deep Cut

        The `calc_parallel_deep_cut` function calculates various values based on the input parameters and returns them as a tuple.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float representing the square of the value tau
        :type tsq: float
        :return: a tuple with three elements.

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
            >>> calc.calc_parallel_deep_cut_old(0.11, 0.01, 0.01)
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
