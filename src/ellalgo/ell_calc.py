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

    def __init__(self, n: float) -> None:
        self.n_f = n
        self.half_n = n / 2.0
        self.cst0 = 1.0 / (n + 1.0)
        self.cst1 = n**2 / (n**2 - 1.0)
        self.cst2 = 2.0 * self.cst0
        self.cst3 = n * self.cst0

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
    def calc_ll(self, beta) -> CutStatus:
        """parallel or deep cut

        Arguments:
            beta ([type]): [description]

        Returns:
            int: [description]
        """
        if np.isscalar(beta):
            return self.calc_dc(beta, sqrt(self.tsq))
        elif len(beta) < 2:  # unlikely
            return self.calc_dc(beta[0], sqrt(self.tsq))
        return self.calc_ll_core(beta[0], beta[1])

    def calc_ll_core(self, b0: float, b1: float) -> CutStatus:
        if b1 < b0:
            return CutStatus.NoSoln  # no sol'n
        if (tau := sqrt(self.tsq)) < b1 or not self.use_parallel_cut:
            return self.calc_dc(b0, tau)
        if b0 == 0.0:
            return self.calc_ll_cc(b1)
        b0b1 = b0 * b1
        if self.n_f * b0b1 < -self.tsq:  # for discrete optimization
            return CutStatus.NoEffect  # no effect

        b0sq = b0**2
        b1sq = b1**2
        t0 = self.tsq - b0sq
        t1 = self.tsq - b1sq
        bsum = b0 + b1
        xi = sqrt(t0 * t1 + (self.half_n * (b1sq - b0sq)) ** 2)
        self.sigma = self.cst3 + self.cst2 * (self.tsq - b0b1 - xi) / (bsum**2)
        self.rho = self.sigma * bsum / 2
        self.delta = self.cst1 * ((t0 + t1) / 2 + xi / self.n_f) / self.tsq
        return CutStatus.Success

    def calc_ll_cc(self, b1: float) -> CutStatus:
        b1sq = b1**2
        xi = sqrt((self.tsq - b1sq) * self.tsq + (self.half_n * b1sq) ** 2)
        self.sigma = self.cst3 + self.cst2 * (self.tsq - xi) / b1sq
        self.rho = self.sigma * b1 / 2
        temp = self.tsq - b1sq / 2 + xi / self.n_f
        self.delta = self.cst1 * temp / self.tsq
        return CutStatus.Success

    def calc_dc(self, beta: float, tau: float) -> CutStatus:
        if tau < beta:
            return CutStatus.NoSoln  # no sol'n
        if beta == 0.0:
            return self.calc_cc(tau)
        gamma = tau + self.n_f * beta
        if gamma < 0.0:
            return CutStatus.NoEffect  # no effect

        self.rho = self.cst0 * gamma
        self.sigma = self.cst2 * gamma / (tau + beta)
        self.delta = self.cst1 * (self.tsq - beta**2) / self.tsq
        return CutStatus.Success

    def calc_cc(self, tau: float) -> CutStatus:
        self.sigma = self.cst2
        self.rho = self.cst0 * tau
        self.delta = self.cst1
        return CutStatus.Success

    def get_results(self) -> List[float]:
        return [self.rho, self.sigma, self.delta, self.tsq]


# trait UpdateByCutChoices:
#     def update_by(self, ell: &mut EllCalc) -> CutStatus
#
