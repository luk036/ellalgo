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
        """
        The function initializes several variables based on the input value.

        :param n_f: The parameter `n_f` represents a floating point value. It is used to
        initialize the `EllCalcCore` object
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
        self._n_sq = self._n_f * self._n_f
        self._cst0 = 1.0 / self._n_plus_1
        self._cst1 = self._n_sq / (self._n_sq - 1.0)
        self._cst2 = 2.0 * self._cst0
        self._cst3 = self._n_f * self._cst0

    #                  2
    #            σ = ─────
    #                n + 1
    #
    #                  τ
    #            ϱ = ─────
    #                n + 1
    #
    #                   2
    #                  n
    #            δ = ──────
    #                 2
    #                n  - 1
    #
    def calc_central_cut(self, tau: float) -> Tuple[float, float, float]:
        """Calculate Central Cut

        The `calc_central_cut` function calculates the central cut values based on the given input.

        :param tau: tau is a float representing the value of tau
        :type tau: float
        :return: The function `calc_central_cut` returns a tuple containing the following elements:

        Examples:
            >>> calc = EllCalcCore(3)
            >>> calc.calc_central_cut(4.0)
            (1.0, 0.5, 1.125)
        """
        rho = self._cst0 * tau
        sigma = self._cst2
        delta = self._cst1
        return (rho, sigma, delta)

    #             γ = τ + n ⋅ β
    #
    #                   γ
    #             ϱ = ─────
    #                 n + 1
    #
    #                 2 ⋅ ϱ
    #             σ = ─────
    #                 τ + β
    #
    #                  2   ⎛ 2    2⎞
    #                 n  ⋅ ⎝τ  - β ⎠
    #             δ = ──────────────
    #                  ⎛ 2    ⎞    2
    #                  ⎝n  - 1⎠ ⋅ τ
    #
    def calc_deep_cut(self, beta: float, tau: float) -> Tuple[float, float, float]:
        """Calculate Deep Cut

        The `calc_deep_cut` function calculates the deep cut values based on the given input.

        :param beta: beta is a float representing the value of beta
        :type beta: float
        :param tau: tau is a float representing the value of tau
        :type tau: float
        :return: The function `calc_deep_cut` returns a tuple containing the following elements:

        Examples:
            >>> calc = EllCalcCore(3)
            >>> calc.calc_deep_cut(1.0, 2.0)
            (1.25, 0.8333333333333334, 0.84375)
            >>> calc.calc_deep_cut(0.0, 2.0)
            (0.5, 0.5, 1.125)
        """
        gamma = tau + self._n_f * beta
        alpha = beta / tau
        rho = self._cst0 * gamma
        sigma = self._cst2 * gamma / (tau + beta)
        delta = self._cst1 * (1.0 - alpha) * (1 + alpha)
        return (rho, sigma, delta)

    def calc_parallel_central_cut(
        self, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        """Calculate Parallel Central Cut

        The function `calc_parallel_central_cut` calculates the parallel central cut for given values of `beta1` and
        `tsq`.

        :param beta1: The parameter `beta1` represents a float value. It is used in the calculation of the
        central cut
        :type beta1: float
        :param tsq: The parameter `tsq` represents the square of a value
        :type tsq: float
        :return: The function `calc_parallel_central_cut` returns a tuple of four values: `CutStatus`, `float`,
        `float`, `float`.

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_central_cut(0.11, 0.01)
            (0.01897790039191521, 0.3450527343984584, 1.0549907942519101)
        """
        b1sq = beta1 * beta1
        a1sq = b1sq / tsq
        temp = self._half_n * a1sq
        mu_plus_1 = temp + sqrt(1.0 - a1sq + temp * temp)
        mu_plus_2 = mu_plus_1 + 1.0
        rho = beta1 / mu_plus_2
        sigma = 2.0 / mu_plus_2
        temp2 = self._n_f * mu_plus_1
        delta = temp2 / (temp2 - 1.0)
        return (rho, sigma, delta)

    #                        __________________________
    #                       ╱                         2
    #                      ╱                  ⎛     2⎞
    #                     ╱                   ⎜n ⋅ β ⎟
    #                    ╱   ⎛ 2    2⎞    2   ⎜     1⎟
    #             ξ =   ╱    ⎜τ  - β ⎟ ⋅ τ  + ⎜──────⎟
    #                 ╲╱     ⎝      1⎠        ⎝   2  ⎠
    #
    #                             ⎛ 2    ⎞
    #                   n     2 ⋅ ⎝τ  - ξ⎠
    #             σ = ───── + ────────────
    #                 n + 1              2
    #                         (n + 1) ⋅ β
    #                                    1
    #
    #                 σ ⋅ β
    #                      1
    #             ϱ = ──────
    #                    2
    #
    #                      ⎛      2    ⎞
    #                      ⎜     β     ⎟
    #                  2   ⎜ 2    1   ξ⎟
    #                 n  ⋅ ⎜τ  - ── + ─⎟
    #                      ⎝      2   n⎠
    #             δ = ──────────────────
    #                    ⎛ 2    ⎞    2
    #                    ⎝n  - 1⎠ ⋅ τ
    #
    def calc_parallel_central_cut_old(
        self, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        """Calculate Parallel Central Cut

        The function `calc_parallel_central_cut` calculates the parallel central cut for given values of `beta1` and
        `tsq`.

        :param beta1: The parameter `beta1` represents a float value. It is used in the calculation of the
        central cut
        :type beta1: float
        :param tsq: The parameter `tsq` represents the square of a value
        :type tsq: float
        :return: The function `calc_parallel_central_cut` returns a tuple of four values: `CutStatus`, `float`,
        `float`, `float`.

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_central_cut_old(0.11, 0.01)
            (0.018977900391915218, 0.3450527343984585, 1.0549907942519101)
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
        """Calculation Parallel Deep Cut

        The `calc_parallel_deep_cut` function calculates various values based on the input parameters and returns
        them as a tuple.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float representing the square of the value t
        :type tsq: float
        :return: a tuple with three elements.

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_deep_cut(0.11, 0.01, 0.01)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_deep_cut(-0.25, 0.25, 1.0)
            (0.0, 0.8, 1.25)
        """
        b0b1 = beta0 * beta1
        bsum = beta0 + beta1
        bsumsq = bsum * bsum
        temp = self._half_n * bsumsq
        a = tsq + self._n_f * b0b1
        h = tsq + b0b1 + self._half_n * bsumsq
        c = self._n_plus_1 * bsumsq  # (n + 1) * bsumsq
        temp2 = h + sqrt(h * h - a * c)
        inv_mu_plus_2 = a / temp2
        # mu = temp2 / a - 2.0
        inv_mu = a / (temp2 - 2.0 * a)
        rho = bsum * inv_mu_plus_2
        sigma = 2.0 * inv_mu_plus_2
        delta = 1.0 + (-2.0 * b0b1 + bsumsq * inv_mu_plus_2) * inv_mu / tsq
        return (rho, sigma, delta)

    #                  2    2
    #            ζ  = τ  - β
    #             0         0
    #
    #                  2    2
    #            ζ  = τ  - β
    #             1         1
    #                       __________________________
    #                      ╱                         2
    #                     ╱           ⎛    ⎛ 2    2⎞⎞
    #                    ╱            ⎜n ⋅ ⎜β  - β ⎟⎟
    #                   ╱             ⎜    ⎝ 1    0⎠⎟
    #            ξ =   ╱    ζ  ⋅ ζ  + ⎜─────────────⎟
    #                ╲╱      0    1   ⎝      2      ⎠
    #
    #                            ⎛ 2              ⎞
    #                        2 ⋅ ⎜τ  + β  ⋅ β  - ξ⎟
    #                  n         ⎝      0    1    ⎠
    #            σ = ───── + ──────────────────────
    #                n + 1                       2
    #                         (n + 1) ⋅ ⎛β  + β ⎞
    #                                   ⎝ 0    1⎠
    #
    #                σ ⋅ ⎛β  + β ⎞
    #                    ⎝ 0    1⎠
    #            ϱ = ─────────────
    #                      2
    #
    #                     ⎛ζ  + ζ     ⎞
    #                 2   ⎜ 0    1   ξ⎟
    #                n  ⋅ ⎜─────── + ─⎟
    #                     ⎝   2      n⎠
    #            δ = ──────────────────
    #                   ⎛ 2    ⎞    2
    #                   ⎝n  - 1⎠ ⋅ τ
    #
    def calc_parallel_deep_cut_old(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        """Calculation Parallel Deep Cut

        The `calc_parallel_deep_cut` function calculates various values based on the input parameters and returns
        them as a tuple.

        :param beta0: The parameter `beta0` represents a float value
        :type beta0: float
        :param beta1: The parameter `beta1` represents a float value
        :type beta1: float
        :param tsq: tsq is a float representing the square of the value t
        :type tsq: float
        :return: a tuple with three elements.

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_deep_cut_old(0.11, 0.01, 0.01)
            (0.02722850906828212, 0.4538084844713687, 1.0443438549074862)
        """
        b0b1 = beta0 * beta1
        b0sq = beta0 * beta0
        b1sq = beta1 * beta1
        t0 = tsq - b0sq
        t1 = tsq - b1sq
        xi = sqrt(t0 * t1 + (self._half_n * (b1sq - b0sq)) ** 2)
        bsumsq = (beta0 + beta1) ** 2
        sigma = self._cst3 + self._cst2 * (tsq + b0b1 - xi) / bsumsq
        rho = sigma * (beta0 + beta1) / 2.0
        delta = self._cst1 * ((t0 + t1) / 2.0 + xi / self._n_f) / tsq
        return (rho, sigma, delta)
