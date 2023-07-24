from .ell_config import CutStatus
from math import sqrt
from typing import Tuple


class EllCalc:
    """_summary_

    Examples:
        >>> from ellalgo.ell_calc import EllCalc
        >>> calc = EllCalc(3)   
    """
    use_parallel_cut: bool = True

    _n_f: float
    _half_n: float
    _cst0: float
    _cst1: float
    _cst2: float
    _cst3: float

    def __init__(self, n: int) -> None:
        """_summary_

        Args:
            n (int): _description_

        Examples:
            >>> from ellalgo.ell_calc import EllCalc
            >>> calc = EllCalc(3)   
        """
        self._n_f = float(n)
        self._half_n = self._n_f / 2.0
        self._cst0 = 1.0 / (self._n_f + 1.0)
        self._cst1 = self._n_f**2 / (self._n_f**2 - 1.0)
        self._cst2 = 2.0 * self._cst0
        self._cst3 = self._n_f * self._cst0

    def calc_single_or_ll(
        self, beta, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """single deep cut or parallel cut

        Args:
            beta (_type_): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        if isinstance(beta, (int, float)):
            return self.calc_dc(beta, tsq)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_dc(beta[0], tsq)
        return self.calc_ll(beta[0], beta[1], tsq)

    def calc_single_or_ll_cc(
        self, beta, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """single central cut or parallel cut

        Args:
            beta (_type_): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        if isinstance(beta, (int, float)) or len(beta) < 2 or not self.use_parallel_cut:
            return self.calc_cc(tsq)
        return self.calc_ll_cc(beta[1], tsq)

    def calc_ll(
        self, b0: float, b1: float, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """parallel deep cut

             ⎛                      ╱     ╱    ⎞
            -τ                0    β0    β1    +τ
             ⎝                    ╱     ╱      ⎠

        Args:
            b0 (float): _description_
            b1 (float): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        if b1 < b0:
            return (CutStatus.NoSoln, 0.0, 0.0, 0.0)  # no sol'n
        # if b0 == 0.0:
        #     return self.calc_ll_cc(b1)
        b1sq = b1 * b1
        if b1 > 0.0 and tsq < b1sq:
            return self.calc_dc(b0, tsq)
        b0b1 = b0 * b1
        return self.calc_ll_core(b0, b1, b1sq, b0b1, tsq)

    def calc_ll_core(
        self, b0: float, b1: float, b1sq: float, b0b1: float, tsq
    ) -> Tuple[CutStatus, float, float, float]:
        """Parallel deep cut core

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
                         (n + 1) ⋅ ⎛β  + β ⎞
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

        Args:
            b0 (float): _description_
            b1 (float): _description_
            b0sq (float): _description_
            b0b1 (float): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        b0sq = b0 * b0
        t0 = tsq - b0sq
        t1 = tsq - b1sq
        xi = sqrt(t0 * t1 + (self._half_n * (b1sq - b0sq)) ** 2)
        bsumsq = b0sq + 2.0 * b0b1 + b1sq
        sigma = self._cst3 + self._cst2 * (tsq + b0b1 - xi) / bsumsq
        rho = sigma * (b0 + b1) / 2.0
        delta = self._cst1 * ((t0 + t1) / 2.0 + xi / self._n_f) / tsq
        return (CutStatus.Success, rho, sigma, delta)

    def calc_ll_cc(
        self, b1: float, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """Parallel central cut
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

        Args:
            b1 (float): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        if b1 < 0.0:
            return (CutStatus.NoSoln, 0.0, 0.0, 0.0)  # no sol'n
        b1sq = b1 * b1
        if tsq < b1sq or not self.use_parallel_cut:
            return self.calc_cc(tsq)
        # Core calculation
        a1sq = b1sq / tsq
        xi = sqrt(1.0 - a1sq + (self._half_n * a1sq) ** 2)
        sigma = self._cst3 + self._cst2 * (1.0 - xi) / a1sq
        rho = sigma * b1 / 2.0
        # temp = 1.0 - a1sq / 2 + xi / self._n_f
        delta = self._cst1 * (1.0 - a1sq / 2.0 + xi / self._n_f)
        return (CutStatus.Success, rho, sigma, delta)

    def calc_dc(self, beta: float, tsq: float) -> Tuple[CutStatus, float, float, float]:
        """Deep Cut

            γ = τ + n ⋅ β

                  γ
            ϱ = ─────
                n + 1

                2 ⋅ ϱ
            σ = ─────
                τ + β

                 2   ⎛ 2    2⎞
                n  ⋅ ⎝τ  - β ⎠
            δ = ──────────────
                 ⎛ 2    ⎞    2
                 ⎝n  - 1⎠ ⋅ τ

        Args:
            beta (float): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        assert beta >= 0.0
        bsq = beta * beta
        if tsq < bsq:
            return (CutStatus.NoSoln, 0.0, 0.0, 0.0)  # no sol'n
        tau = sqrt(tsq)
        gamma = tau + self._n_f * beta
        return self.calc_dc_core(beta, tau, gamma)

    def calc_dc_core(
        self, beta: float, tau: float, gamma: float
    ) -> Tuple[CutStatus, float, float, float]:
        """Deep cut core

            γ = τ + n ⋅ β

                  γ
            ϱ = ─────
                n + 1

                2 ⋅ ϱ
            σ = ─────
                τ + β

                 2   ⎛ 2    2⎞
                n  ⋅ ⎝τ  - β ⎠
            δ = ──────────────
                 ⎛ 2    ⎞    2
                 ⎝n  - 1⎠ ⋅ τ

        Args:
            beta (float): _description_
            tau (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        rho = self._cst0 * gamma
        sigma = self._cst2 * gamma / (tau + beta)
        delta = self._cst1 * (1.0 - (beta / tau) ** 2)
        return (CutStatus.Success, rho, sigma, delta)

    def calc_cc(self, tsq: float) -> Tuple[CutStatus, float, float, float]:
        """Central Cut

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

        Args:
            tau (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_

        Examples:
        """
        sigma = self._cst2
        rho = self._cst0 * sqrt(tsq)
        delta = self._cst1
        return (CutStatus.Success, rho, sigma, delta)

    def calc_single_or_ll_q(
        self, beta, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """single deep cut or parallel cut (discrete)

        Args:
            beta ([type]): [description]
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        if isinstance(beta, (int, float)):
            return self.calc_dc_q(beta, tsq)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_dc_q(beta[0], tsq)
        return self.calc_ll_q(beta[0], beta[1], tsq)

    def calc_ll_q(
        self, b0: float, b1: float, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """Parallel deep cut (discrete)

             ⎛                      ╱     ╱    ⎞
            -τ                0    β0    β1    +τ
             ⎝                    ╱     ╱      ⎠

        Args:
            b0 (float): _description_
            b1 (float): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        if b1 < b0:
            return (CutStatus.NoSoln, 0.0, 0.0, 0.0)  # no sol'n
        # if b0 == 0.0:
        #     return self.calc_ll_cc(b1)
        b1sq = b1 * b1
        if b1 > 0.0 and tsq < b1sq:
            return self.calc_dc_q(b0, tsq)
        b0b1 = b0 * b1
        if self._n_f * b0b1 < -tsq:  # for discrete optimization
            return (CutStatus.NoEffect, 0.0, 0.0, 0.0)  # no effect
        # TODO: check b0 + b1 == 0
        return self.calc_ll_core(b0, b1, b1sq, b0b1, tsq)

    def calc_dc_q(
        self, beta: float, tsq: float
    ) -> Tuple[CutStatus, float, float, float]:
        """Deep Cut (discrete)

            γ = τ + n ⋅ β

                  γ
            ϱ = ─────
                n + 1

                2 ⋅ ϱ
            σ = ─────
                τ + β

                 2   ⎛ 2    2⎞
                n  ⋅ ⎝τ  - β ⎠
            δ = ──────────────
                 ⎛ 2    ⎞    2
                 ⎝n  - 1⎠ ⋅ τ

        Args:
            beta (float): _description_
            tsq (float): _description_

        Returns:
            Tuple[CutStatus, float, float, float]: _description_
        """
        tau = sqrt(tsq)
        if tau < beta:
            return (CutStatus.NoSoln, 0.0, 0.0, 0.0)  # no sol'n
        gamma = tau + self._n_f * beta
        if gamma <= 0.0:
            return (CutStatus.NoEffect, 0.0, 0.0, 0.0)
        return self.calc_dc_core(beta, tau, gamma)


if __name__ == "__main__":
    from pytest import approx

    ell_calc = EllCalc(4)
    status, rho, sigma, delta = ell_calc.calc_ll_q(0.07, 0.03, 0.01)
    assert status == CutStatus.NoSoln

    status, rho, sigma, delta = ell_calc.calc_ll_q(0.0, 0.05, 0.01)
    assert status == (CutStatus.Success, rho, sigma, delta)
    assert sigma == approx(0.8)
    assert rho == approx(0.02)
    assert delta == approx(1.2)

    status, rho, sigma, delta = ell_calc.calc_ll_q(0.05, 0.11, 0.01)
    assert status == (CutStatus.Success, rho, sigma, delta)
    assert sigma == approx(0.8)
    assert rho == approx(0.06)
    assert delta == approx(0.8)

    # status, rho, sigma, delta = ell_calc.calc_ll(-0.07, 0.07)
    # assert status == CutStatus.NoEffect

    status, rho, sigma, delta = ell_calc.calc_ll_q(0.01, 0.04, 0.01)
    assert status == (CutStatus.Success, rho, sigma, delta)
    assert sigma == approx(0.928)
    assert rho == approx(0.0232)
    assert delta == approx(1.232)

    ell_calc = EllCalc(4)
    assert ell_calc.use_parallel_cut is True
    assert ell_calc._n_f == 4.0
    assert ell_calc._half_n == 2.0
    assert ell_calc._cst0 == 0.2
    # assert ell_calc._cst1 == approx(16.0 / 15.0)
    assert ell_calc._cst2 == 0.4
    assert ell_calc._cst3 == 0.8
    print(ell_calc._cst1)
