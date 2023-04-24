from typing import Optional, Tuple

import numpy as np

from .chol_ext import LDLTMgr

Arr = np.ndarray
Cut = Tuple[Arr, float]


class LMIOracle:
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, F, B) -> None:
        """Construct a new lmi oracle object

        Arguments:
            F (List[Arr]): [description]
            B (Arr): [description]
        """
        self.F = F
        self.F0 = B
        self.Q = LDLTMgr(len(B))

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (Arr): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def get_elem(i, j):
            n = len(x)
            return self.F0[i, j] - sum(self.F[k][i, j] * x[k] for k in range(n))

        if self.Q.factor(get_elem):
            return None

        ep = self.Q.witness()
        g = np.array([self.Q.sym_quad(Fk) for Fk in self.F])
        return g, ep
