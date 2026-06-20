"""
Core mathematical computations for ellipsoid updates.

The `EllCalcCore` class implements the low-level formulas that compute the
ellipsoid update parameters (rho, sigma, delta) for various types of cuts.
These three parameters define the transformation applied to the ellipsoid's
center and shape matrix after each cutting-plane iteration.

Supported cut calculations:
    - calc_central_cut: Central cut through the ellipsoid center.
    - calc_bias_cut / calc_bias_cut_fast: Deep (non-central) cuts.
    - calc_parallel_central_cut: Parallel central cut.
    - calc_parallel_bias_cut / calc_parallel_bias_cut_fast: Parallel deep cuts.
    - calc_parallel_bias_cut_fast2 / _old variants: Alternative formulations.

All methods return (rho, sigma, delta) — the center displacement, shape scaling,
and size adjustment factor respectively.
"""

from math import sqrt
from typing import Tuple


class EllCalcCore:
    """
    The `EllCalcCore` class provides the low-level mathematical calculations
    required for the ellipsoid method. It computes the essential parameters (rho,
    sigma, and delta) that define the transformation of the search ellipsoid at
    each iteration of the algorithm.

    - `rho` (ρ): Represents the displacement of the ellipsoid's center.
    - `sigma` (σ): A scaling factor for the ellipsoid's shape matrix.
    - `delta` (δ): A factor that adjusts the overall size of the ellipsoid.

    This class is designed to be used by `EllCalc`, which handles the higher-level
    logic of the ellipsoid method. By encapsulating the core calculations,
    `EllCalcCore` allows for a clean separation of concerns and facilitates
    maintenance and optimization of the mathematical formulas.
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

        Precomputes frequently used constants for ellipsoid calculations:
        - _half_n: Half of the dimension n_f
        - _n_plus_1: n_f + 1
        - _inv_n: 1/n_f
        - _n_sq: n_f squared
        - _cst0 to _cst3: Various constants used in cut calculations

        Args:
            n_f: The dimension of the space.

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
        r"""Update parameters for a central cut.

        A central cut passes through the ellipsoid center. The update
        parameters :math:`\rho, \sigma, \delta` are dimension-dependent
        constants scaled by :math:`\tau`:

        .. math::

           \rho &= \frac{1}{n+1}\,\tau \\[4pt]
           \sigma &= \frac{2}{n+1} \\[4pt]
           \delta &= \frac{n^2}{n^2-1}

        where :math:`n` is the dimension of the space and :math:`\tau`
        is the distance from the center to the ellipsoid boundary along
        the cut direction.

        Args:
            tau: The distance parameter for the central cut.

        Returns:
            Tuple (rho, sigma, delta) for the ellipsoid update.
        """
        rho = self._cst0 * tau
        sigma = self._cst2
        delta = self._cst1
        return (rho, sigma, delta)

    def calc_bias_cut_fast(
        self, beta: float, tau: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Deep cut ellipsoid parameters (with precomputed :math:`\eta`).

        A deep (non-central) cut at distance :math:`\beta` from the center.
        The auxiliary variable :math:`\eta = \tau + n \beta` is passed in
        to avoid recomputation.

        .. math::

           \eta &= \tau + n \beta \\[4pt]
           \rho &= \frac{\eta}{n+1} \\[4pt]
           \sigma &= \frac{2\rho}{\tau + \beta}
                 = \frac{2\eta}{(n+1)(\tau + \beta)} \\[4pt]
           \delta &= \frac{n^2}{n^2-1}\,
                    \frac{\tau^2 - \beta^2}{\tau^2}

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

        :param beta: Bias distance from center to cut hyperplane
        :param tau: Distance parameter for the cut
        :param eta: Precomputed :math:`\tau + n \beta`
        :return: Tuple :math:`(\rho, \sigma, \delta)` for the biased cut

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
        r"""Deep cut ellipsoid parameters (standard version).

        Deep cut where the hyperplane does **not** pass through the center.
        Computes :math:`\eta = \tau + n \beta` internally and delegates to
        :meth:`calc_bias_cut_fast`.

        .. math::

           \eta &= \tau + n \beta \\[4pt]
           \rho &= \frac{\eta}{n+1} \\[4pt]
           \sigma &= \frac{2\eta}{(n+1)(\tau + \beta)} \\[4pt]
           \delta &= \frac{n^2}{n^2-1}\,
                    \frac{\tau^2 - \beta^2}{\tau^2}

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

        :param beta: Bias distance from center to cut hyperplane
        :param tau: Distance parameter for the cut
        :return: Tuple :math:`(\rho, \sigma, \delta)` for the biased cut

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
        r"""Parallel central cut parameters.

        A cut parallel to a central cut, offset by :math:`\beta_1`.
        The auxiliary variable :math:`\alpha = \beta_1 / \tau` and
        :math:`k = \frac{n}{2}\alpha^2` define the root :math:`r`:

        .. math::

           \alpha^2 &= \frac{\beta_1^2}{\tau^2} \\[4pt]
           k &= \frac{n}{2}\,\alpha^2 \\[4pt]
           r &= k + \sqrt{k^2 + 1 - \alpha^2} \\[4pt]
           \rho &= \frac{\beta_1}{r+1} \\[4pt]
           \sigma &= \frac{2}{r+1} \\[4pt]
           \delta &= \frac{r}{r - 1/n}

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

        :param beta1: Offset parameter for the parallel cut
        :param tsq: Square of the distance parameter :math:`\tau^2`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

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

        :param beta1: Offset parameter for the parallel cut
        :param tsq: Square of :math:`\tau`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

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
        r"""Parallel deep cut (standard version).

        Two parallel biased cuts at :math:`\beta_0` and :math:`\beta_1`.
        Computes intermediates :math:`\eta = \tau^2 + n \beta_0 \beta_1`
        and delegates to :meth:`calc_parallel_bias_cut_fast`.

        .. math::

           \eta &= \tau^2 + n\,\beta_0 \beta_1 \\[4pt]
           \bar\beta &= \frac{\beta_0 + \beta_1}{2} \\[4pt]
           h &= \tfrac12\bigl(\tau^2 + \beta_0 \beta_1\bigr) +
                n\,\bar\beta^{\,2} \\[4pt]
           k &= h + \sqrt{h^2 - (n+1)\,\eta\,\bar\beta^{\,2}} \\[4pt]
           \sigma &= \frac{\eta}{k} \\[4pt]
           \rho &= \bar\beta\,\sigma \\[4pt]
           \frac{1}{\mu} &= \frac{\eta}{k - \eta} \\[4pt]
           \delta &= 1 + \frac{1}{\mu\tau^2}\,
                    \bigl(\bar\beta^{\,2}\sigma - \beta_0\beta_1\bigr)

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

        :param beta0: First bias parameter
        :param beta1: Second bias parameter
        :param tsq: Square of :math:`\tau`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

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

    def calc_parallel_bias_cut_fast_old(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Parallel deep cut (original fast formulation).

        Uses :math:`\bar\beta = (\beta_0+\beta_1)/2` and the same
        :math:`h, k` computation as the standard version, but expressed
        as :math:`\sigma = 1/(\mu+1)`.

        .. math::

           \bar\beta &= \frac{\beta_0 + \beta_1}{2} \\[4pt]
           h &= \frac12(\tau^2 + \beta_0\beta_1) + n\,\bar\beta^{\,2} \\[4pt]
           k &= h + \sqrt{h^2 - (n+1)\,\eta\,\bar\beta^{\,2}} \\[4pt]
           \sigma &= \frac{\eta}{k} \\[4pt]
           \rho &= \bar\beta\,\sigma \\[4pt]
           \frac{1}{\mu} &= \frac{\eta}{k-\eta} \\[4pt]
           \delta &= 1 + \frac{1}{\mu\tau^2}\,
                    \bigl(\bar\beta^{\,2}\sigma - \beta_0\beta_1\bigr)

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

        :param beta0: First bias parameter
        :param beta1: Second bias parameter
        :param tsq: Square of :math:`\tau`
        :param b0b1: Precomputed :math:`\beta_0\beta_1`
        :param eta: Precomputed :math:`\tau^2 + n\beta_0\beta_1`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCore(4)
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

    def calc_parallel_bias_cut_fast(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Parallel deep cut (optimized formulation).

        Uses auxiliary variables :math:`\zeta_0 = \tau^2 - \beta_0^2` and
        :math:`\zeta_1 = \tau^2 - \beta_1^2` to compute:

        .. math::

           \zeta_0 &= \tau^2 - \beta_0^2 \\[4pt]
           \zeta_1 &= \tau^2 - \beta_1^2 \\[4pt]
           \xi &= \sqrt{\zeta_0 \zeta_1 +
                        \left(\frac{n}{2}(\beta_1^2 - \beta_0^2)\right)^{\!2}} \\[6pt]
           \sigma &= \frac{2\eta}{\tau^2 + \beta_0\beta_1 +
                        \frac{n}{2}(\beta_0+\beta_1)^2 + \xi} \\[4pt]
           \rho &= \sigma\,\frac{\beta_0 + \beta_1}{2} \\[4pt]
           \delta &= \frac{n^2}{(n^2-1)\,\tau^2}\,
                    \left(\frac{\zeta_0 + \zeta_1}{2} + \frac{\xi}{n}\right)

        :param beta0: First bias parameter
        :param beta1: Second bias parameter
        :param tsq: Square of :math:`\tau`
        :param b0b1: Precomputed :math:`\beta_0\beta_1`
        :param eta: Precomputed :math:`\tau^2 + n\beta_0\beta_1`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCore(4)
            >>> calc.calc_parallel_bias_cut_fast(0.11, 0.01, 0.01, 0.0011, 0.0144)
            (0.027228509068282114, 0.45380848447136857, 1.0443438549074862)
            >>> calc.calc_parallel_bias_cut_fast(-0.25, 0.25, 1.0, -0.0625, 0.75)
            (0.0, 0.8, 1.25)
        """
        b0sq = beta0 * beta0
        b1sq = beta1 * beta1
        zeta0 = tsq - b0sq
        zeta1 = tsq - b1sq
        xi = sqrt(zeta0 * zeta1 + (self._half_n * (b1sq - b0sq)) ** 2)
        bsumsq = (beta0 + beta1) ** 2
        # sigma = self._cst3 + self._cst2 * (tsq + b0b1 - xi) / bsumsq
        sigma = 2.0 * eta / (tsq + b0b1 + self._half_n * bsumsq + xi)
        rho = sigma * (beta0 + beta1) / 2.0
        delta = self._cst1 * ((zeta0 + zeta1) / 2.0 + xi / self._n_f) / tsq
        return (rho, sigma, delta)

    def calc_parallel_bias_cut_fast2(
        self, beta0: float, beta1: float, tsq: float, b0b1: float, eta: float
    ) -> Tuple[float, float, float]:
        r"""Parallel deep cut (alternative sigma formulation).

        Same :math:`\zeta` and :math:`\xi` as :meth:`calc_parallel_bias_cut_fast`
        but uses an alternative :math:`\sigma` formula:

        .. math::

           \zeta_0 &= \tau^2 - \beta_0^2 \\[4pt]
           \zeta_1 &= \tau^2 - \beta_1^2 \\[4pt]
           \xi &= \sqrt{\zeta_0 \zeta_1 +
                        \left(\frac{n}{2}(\beta_1^2 - \beta_0^2)\right)^{\!2}} \\[6pt]
           \sigma &= \frac{n}{n+1} +
                    \frac{2(\tau^2 + \beta_0\beta_1 - \xi)}
                         {(n+1)(\beta_0+\beta_1)^2}

        The :math:`\rho` and :math:`\delta` formulas are identical to
        :meth:`calc_parallel_bias_cut_fast`.

        Args:
            beta0: First bias parameter
            beta1: Second bias parameter
            tsq: Square of :math:`\tau`
            b0b1: Precomputed :math:`\beta_0\beta_1`
            eta: Precomputed :math:`\tau^2 + n\beta_0\beta_1`

        Returns:
            Tuple :math:`(\rho, \sigma, \delta)`

        Examples:
            >>> calc = EllCalcCore(4)
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
        # sigma = 2.0 * eta / (tsq + b0b1 + self._n_f * bsumsq + xi)
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

        .. math::

           \zeta_0 &= \tau^2 - \beta_0^2 \\[4pt]
           \zeta_1 &= \tau^2 - \beta_1^2 \\[4pt]
           \xi &= \sqrt{\zeta_0\zeta_1 +
                        \left(\frac{n(\beta_1^2 - \beta_0^2)}{2}\right)^{\!2}} \\[6pt]
           \sigma &= \frac{n}{n+1} +
                    \frac{2(\tau^2 + \beta_0\beta_1 - \xi)}
                         {(n+1)(\beta_0 + \beta_1)^2} \\[4pt]
           \rho &= \frac{\sigma(\beta_0 + \beta_1)}{2} \\[4pt]
           \delta &= \frac{n^2}{(n^2-1)\tau^2}\,
                    \left(\frac{\zeta_0 + \zeta_1}{2} + \frac{\xi}{n}\right)

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

        :param beta0: First bias parameter
        :param beta1: Second bias parameter
        :param tsq: Square of :math:`\tau`
        :return: Tuple :math:`(\rho, \sigma, \delta)`

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
