from math import sqrt
from typing import List

import numpy as np

from .cutting_plane import CutStatus


class EllCalc:
    use_parallel_cut: bool = True
    rho: float = 0.0
    sigma: float = 0.0
    delta: float = 0.0
    tsq: float = 0.0
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
        E = EllCalc(self.n_f)
        E.use_parallel_cut = self.use_parallel_cut
        E.rho = self.rho
        E.sigma = self.sigma
        E.delta = self.delta
        E.tsq = self.tsq
        return E

    # def update_cut(self, beta: float) -> CutStatus { self.calc_dc(beta)
    def calc_single_or_ll(self, beta) -> CutStatus:
        """single or parallel cut

        Args:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if np.isscalar(beta):
            return self.calc_dc(beta)
        elif len(beta) < 2:  # unlikely
            return self.calc_dc(beta[0])
        return self.calc_ll(beta[0], beta[1])

    def calc_single_or_ll_cc(self, beta) -> CutStatus:
        """single or parallel cut

        Args:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if np.isscalar(beta) or len(beta) < 2:
            return self.calc_cc(sqrt(self.tsq))
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
        if b0 == 0.0:
            return self.calc_ll_cc(b1)
        b1sq = b1 * b1
        if self.tsq < b1sq or not self.use_parallel_cut:
            return self.calc_dc(b0)
        b0b1 = b0 * b1
        if self.n_f * b0b1 < -self.tsq:  # for discrete optimization
            return CutStatus.NoEffect  # no effect
        self.calc_ll_core(b0, b1, b1sq, b0b1)
        return CutStatus.Success

    def calc_ll_core(self, b0: float, b1: float, b1sq: float, b0b1: float) -> None:
        """Parallel Cut Core

                  2    2
            ζ  = τ  - β
             0         0

                  2    2
            ζ  = τ  - β
             1         0
                       __________________________
                      ╱                         2
                     ╱           ⎛    ⎛ 2    2⎞⎞
                    ╱            ⎜n ⋅ ⎜β  - β ⎟⎟
                   ╱             ⎜    ⎝ 1    0⎠⎟
            ξ =   ╱    ζ  ⋅ ζ  + ⎜─────────────⎟
                ╲╱      0    1   ⎝      2      ⎠

                            ⎛ 2              ⎞
                        2 ⋅ ⎜τ  - β  ⋅ β  - ξ⎟
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
        self.sigma = self.cst3 + self.cst2 * (self.tsq - b0b1 - xi) / bsumsq
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
        b1sq = b1**2
        if self.tsq < b1sq or not self.use_parallel_cut:
            return self.calc_cc(sqrt(self.tsq))
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
            tau (float): _description_

        Returns:
            CutStatus: _description_
        """
        tau = sqrt(self.tsq)
        if tau < beta:
            return CutStatus.NoSoln  # no sol'n
        if beta == 0.0:
            return self.calc_cc(tau)
        gamma = tau + self.n_f * beta

        if gamma < 0.0:  # discrete optimization only
            return CutStatus.NoEffect  # no effect

        self.rho = self.cst0 * gamma
        self.sigma = self.cst2 * gamma / (tau + beta)
        self.delta = self.cst1 * (self.tsq - beta**2) / self.tsq
        return CutStatus.Success

    def calc_cc(self, tau: float) -> CutStatus:
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
        self.rho = self.cst0 * tau
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
