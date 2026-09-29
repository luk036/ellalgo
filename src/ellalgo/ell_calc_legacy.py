"""
Legacy and alternative ellipsoid-update formulations.

These slower / alternative parallel-cut formulas are kept for cross-language
reference (see ``compare-py-cpp-rs.md``) and regression testing only. The
production cut-selection code in :mod:`ellalgo.ell_calc` uses
:class:`~ellalgo.ell_calc_core.EllCalcCore` exclusively.
"""

from math import sqrt
from typing import Tuple

from .ell_calc_core import EllCalcCore


class EllCalcCoreLegacy(EllCalcCore):
    """Reference-only alternative/legacy formulas (not used in production).

    Inherits every production calculation from :class:`EllCalcCore` and adds
    the historical ``*_old``/``*_fast2`` variants.
    """

    def calc_parallel_central_cut_old(
        self, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Parallel central cut (original formulation).

        Original (slower) formulation of the parallel central cut.

        .. math::

           \xi &= \sqrt{(\tau^2 - \beta_1^2)\,\tau^2 +
                        \left(\frac{n \beta_1^2}{2}\right)^{\!2}} \\[6pt]
           \sigma &= \frac{n}{n+1} +
                     \frac{2(\tau^2 - \xi)}{(n+1)\beta_1^2} \\[4pt]
           \rho &= \frac{\sigma \beta_1}{2} \\[4pt]
           \delta &= \frac{n^2}{(n^2-1)\tau^2}\,
                    \left(\tau^2 - \frac{\beta_1^2}{2} + \frac{\xi}{n}\right)

        :param beta1: Offset parameter for the parallel cut
        :param tsq: Square of :math:`\tau`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCoreLegacy(4)
            >>> calc.calc_parallel_central_cut_old(0.09, 0.01)
            (0.02094183648798086, 0.46537414417735246, 1.082031295477563)
        """
        b1sq = beta1 * beta1
        a1sq = b1sq / tsq
        xi = sqrt(1.0 - a1sq + (self._half_n * a1sq) ** 2)
        sigma = self._cst3 + self._cst2 * (1.0 - xi) / a1sq
        rho = sigma * beta1 / 2.0
        delta = self._cst1 * (1.0 - a1sq / 2.0 + xi / self._n_f)
        return (rho, sigma, delta)

    def calc_parallel_bias_cut_fast_old(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Parallel deep cut (original fast formulation).

        Uses :math:`\bar\beta = (\beta_0+\beta_1)/2` and the same
        :math:`h, k` computation as the standard version, but expressed
        as :math:`\sigma = 1/(\mu+1)`.

        :param beta0: First bias parameter
        :param beta1: Second bias parameter
        :param tsq: Square of :math:`\tau`
        :param b0b1: Precomputed :math:`\beta_0\beta_1`
        :param eta: Precomputed :math:`\tau^2 + n\beta_0\beta_1`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCoreLegacy(4)
            >>> calc.calc_parallel_bias_cut_fast_old(0.11, 0.01, 0.01, 0.0011, 0.0144)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_bias_cut_fast_old(-0.25, 0.25, 1.0, -0.0625, 0.75)
            (0.0, 0.8, 1.25)
        """
        bavg = 0.5 * (beta0 + beta1)
        bavgsq = bavg * bavg
        h = 0.5 * (tsq + b0b1) + self._n_f * bavgsq
        gamma_q = h * h - self._n_plus_1 * eta * bavgsq
        if gamma_q < 0.0:
            gamma_q = 0.0
        k = h + sqrt(gamma_q)

        if k <= eta:
            return self.calc_central_cut(sqrt(tsq))

        sigma = eta / k
        inv_mu = eta / (k - eta)
        rho = bavg * sigma
        delta = (tsq + inv_mu * (bavgsq * sigma - b0b1)) / tsq
        return (rho, sigma, delta)

    def calc_parallel_bias_cut_fast2(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Parallel deep cut (alternative sigma formulation).

        Same :math:`\zeta` and :math:`\xi` as
        :meth:`EllCalcCore.calc_parallel_bias_cut_fast` but uses an
        alternative :math:`\sigma` formula.

        Args:
            beta0: First bias parameter
            beta1: Second bias parameter
            tsq: Square of :math:`\tau`
            b0b1: Precomputed :math:`\beta_0\beta_1`
            eta: Precomputed :math:`\tau^2 + n\beta_0\beta_1`

        Returns:
            Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCoreLegacy(4)
            >>> calc.calc_parallel_bias_cut_fast2(0.11, 0.01, 0.01, 0.0011, 0.0144)
            (0.02722850906828212, 0.4538084844713687, 1.0443438549074862)
            >>> calc.calc_parallel_bias_cut_fast2(0.0, 0.09, 0.01, 0.0, 0.01)
            (0.020941836487980856, 0.46537414417735234, 1.082031295477563)
        """
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

    def calc_parallel_bias_cut_old(
        self, beta0: float, beta1: float, tsq: float
    ) -> Tuple[float, float, float]:
        r"""Parallel deep cut (original slower formulation).

        This version shares the same :math:`\zeta_0, \zeta_1, \xi`
        computation as :meth:`calc_parallel_bias_cut_fast2` but the
        formulas are written in an expanded form.

        :param beta0: First bias parameter
        :param beta1: Second bias parameter
        :param tsq: Square of :math:`\tau`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCoreLegacy(4)
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
