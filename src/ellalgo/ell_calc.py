from .ell_config import CutStatus
from math import sqrt
from typing import List


class EllCalc:
    use_parallel_cut: bool = True
    rho: float = 0.0
    sigma: float = 0.0
    delta: float = 0.0
    tsq: float = 0.0
    ndim: int
    n_f: float
    half_n: float
    cst0: float
    cst1: float
    cst2: float
    cst3: float

    def __init__(self, n: int) -> None:
        """_summary_

        Args:
            n (float): _description_
        """
        self.ndim = n
        self.n_f = float(n)
        self.half_n = self.n_f / 2.0
        self.cst0 = 1.0 / (self.n_f + 1.0)
        self.cst1 = self.n_f**2 / (self.n_f**2 - 1.0)
        self.cst2 = 2.0 * self.cst0
        self.cst3 = self.n_f * self.cst0

    def copy(self):
        """[summary]

        Returns:
            EllCalc: [description]
        """
        ellip = EllCalc(self.ndim)
        ellip.use_parallel_cut = self.use_parallel_cut
        ellip.rho = self.rho
        ellip.sigma = self.sigma
        ellip.delta = self.delta
        ellip.tsq = self.tsq
        return ellip

    # def update_cut(self, beta: float) -> CutStatus { self.calc_dc(beta)
    def calc_single_or_ll(self, beta) -> CutStatus:
        """single or parallel cut

        Args:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if isinstance(beta, (int, float)):
            return self.calc_dc(beta)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_dc(beta[0])
        return self.calc_ll(beta[0], beta[1])

    def calc_single_or_ll_cc(self, beta) -> CutStatus:
        """single or parallel cut

        Args:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if isinstance(beta, (int, float)) or len(beta) < 2 or not self.use_parallel_cut:
            return self.calc_cc()
        if beta[1] < 0.0:
            return CutStatus.NoSoln  # no sol'n
        return self.calc_ll_cc(beta[1])

    def calc_ll(self, b0: float, b1: float) -> CutStatus:
        """Parallel Cut

             ⎛                      ╱     ╱    ⎞
            -τ                0    β0    β1    +τ
             ⎝                    ╱     ╱      ⎠

        Args:
            b0 (float): _description_
            b1 (float): _description_

        Returns:
            CutStatus: _description_
        """
        if b1 < b0:
            return CutStatus.NoSoln  # no sol'n
        # if b0 == 0.0:
        #     return self.calc_ll_cc(b1)
        b1sq = b1 * b1
        if b1 > 0.0 and self.tsq < b1sq:
            return self.calc_dc(b0)
        b0b1 = b0 * b1
        self.calc_ll_core(b0, b1, b1sq, b0b1)
        return CutStatus.Success

    def calc_ll_core(self, b0: float, b1: float, b1sq: float, b0b1: float):
        """Parallel Cut Core

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
            b0b1 (float): _description_

        Returns:
            CutStatus: _description_
        """
        b0sq = b0 * b0
        t0 = self.tsq - b0sq
        t1 = self.tsq - b1sq
        xi = sqrt(t0 * t1 + (self.half_n * (b1sq - b0sq)) ** 2)
        bsumsq = b0sq + 2.0 * b0b1 + b1sq
        self.sigma = self.cst3 + self.cst2 * (self.tsq + b0b1 - xi) / bsumsq
        self.rho = self.sigma * (b0 + b1) / 2.0
        self.delta = self.cst1 * ((t0 + t1) / 2.0 + xi / self.n_f) / self.tsq

    def calc_ll_cc(self, b1: float) -> CutStatus:
        """Parallel Cut with beta0 = 0
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

        Returns:
            CutStatus: _description_
        """
        b1sq = b1 * b1
        if self.tsq < b1sq or not self.use_parallel_cut:
            return self.calc_cc()
        a1sq = b1sq / self.tsq
        xi = sqrt(1.0 - a1sq + (self.half_n * a1sq) ** 2)
        self.sigma = self.cst3 + self.cst2 * (1.0 - xi) / a1sq
        self.rho = self.sigma * b1 / 2.0
        # temp = 1.0 - a1sq / 2 + xi / self.n_f
        self.delta = self.cst1 * (1.0 - a1sq / 2.0 + xi / self.n_f)
        return CutStatus.Success

    def calc_dc(self, beta: float) -> CutStatus:
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

        Returns:
            CutStatus: _description_
        """
        assert beta >= 0.0
        bsq = beta * beta
        if self.tsq < bsq:
            return CutStatus.NoSoln  # no sol'n
        tau = sqrt(self.tsq)
        gamma = tau + self.n_f * beta
        self.rho = self.cst0 * gamma
        self.sigma = self.cst2 * gamma / (tau + beta)
        self.delta = self.cst1 * (self.tsq - bsq) / self.tsq
        return CutStatus.Success

    def calc_cc(self) -> CutStatus:
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
            CutStatus: _description_
        """
        self.sigma = self.cst2
        self.rho = self.cst0 * sqrt(self.tsq)
        self.delta = self.cst1
        return CutStatus.Success

    def get_results(self) -> List[float]:
        """_summary_

        Returns:
            List[float]: _description_
        """
        return [self.rho, self.sigma, self.delta, self.tsq]


# trait UpdateByCutChoices:
#     def update_by(self, ell: &mut EllCalc) -> CutStatus
#


class EllCalcQ(EllCalc):
    def calc_single_or_ll(self, beta) -> CutStatus:
        """single or parallel cut
        (override the base class)

        Args:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if isinstance(beta, (int, float)):
            return self.calc_dc_q(beta)
        elif len(beta) < 2 or not self.use_parallel_cut:  # unlikely
            return self.calc_dc_q(beta[0])
        return self.calc_ll_q(beta[0], beta[1])

    def calc_ll_q(self, b0: float, b1: float) -> CutStatus:
        """Parallel Cut

             ⎛                      ╱     ╱    ⎞
            -τ                0    β0    β1    +τ
             ⎝                    ╱     ╱      ⎠

        Args:
            b0 (float): _description_
            b1 (float): _description_

        Returns:
            CutStatus: _description_
        """
        if b1 < b0:
            return CutStatus.NoSoln  # no sol'n
        # if b0 == 0.0:
        #     return self.calc_ll_cc(b1)
        b1sq = b1 * b1
        if b1 > 0.0 and self.tsq < b1sq:
            return self.calc_dc_q(b0)
        b0b1 = b0 * b1
        if self.n_f * b0b1 < -self.tsq:  # for discrete optimization
            return CutStatus.NoEffect  # no effect
        # TODO: check b0 + b1 == 0
        self.calc_ll_core(b0, b1, b1sq, b0b1)
        return CutStatus.Success

    def calc_dc_q(self, beta: float) -> CutStatus:
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

        Returns:
            CutStatus: _description_
        """
        tau = sqrt(self.tsq)
        if tau < beta:
            return CutStatus.NoSoln  # no sol'n
        gamma = tau + self.n_f * beta
        if gamma <= 0.0:
            return CutStatus.NoEffect
        self.rho = self.cst0 * gamma
        self.sigma = self.cst2 * gamma / (tau + beta)
        self.delta = self.cst1 * (self.tsq - beta**2) / self.tsq
        return CutStatus.Success


if __name__ == "__main__":
    from pytest import approx

    ell_calc_q = EllCalcQ(4)
    ell_calc_q.tsq = 0.01
    status = ell_calc_q.calc_ll_q(0.07, 0.03)
    assert status == CutStatus.NoSoln

    status = ell_calc_q.calc_ll_q(0.0, 0.05)
    assert status == CutStatus.Success
    assert ell_calc_q.sigma == approx(0.8)
    assert ell_calc_q.rho == approx(0.02)
    assert ell_calc_q.delta == approx(1.2)

    status = ell_calc_q.calc_ll_q(0.05, 0.11)
    assert status == CutStatus.Success
    assert ell_calc_q.sigma == approx(0.8)
    assert ell_calc_q.rho == approx(0.06)
    assert ell_calc_q.delta == approx(0.8)

    # status = ell_calc.calc_ll(-0.07, 0.07)
    # assert status == CutStatus.NoEffect

    status = ell_calc_q.calc_ll_q(0.01, 0.04)
    assert status == CutStatus.Success
    assert ell_calc_q.sigma == approx(0.928)
    assert ell_calc_q.rho == approx(0.0232)
    assert ell_calc_q.delta == approx(1.232)

    ell_calc = EllCalc(4)
    assert ell_calc.use_parallel_cut is True
    assert ell_calc.n_f == 4.0
    assert ell_calc.half_n == 2.0
    assert ell_calc.cst0 == 0.2
    # assert ell_calc.cst1 == approx(16.0 / 15.0)
    assert ell_calc.cst2 == 0.4
    assert ell_calc.cst3 == 0.8
    print(ell_calc.cst1)
