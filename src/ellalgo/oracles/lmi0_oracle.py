from typing import Optional, Tuple

import numpy as np

from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMI0Oracle:
    """Oracle for Linear Matrix Inequality constraint

    find  x
    s.t.  F * x ⪰ 0

    """

    def __init__(self, F):
        """[summary]

        Arguments:
            F (List[np.ndarray]): [description]
        """
        self.F = F
        self.Q = LDLTMgr(len(F[0]))

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (np.ndarray): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def get_elem(i, j):
            n = len(x)
            return sum(self.F[k][i, j] * x[k] for k in range(n))

        if not self.Q.factor(get_elem):
            ep = self.Q.witness()
            g = np.array([-self.Q.sym_quad(Fk) for Fk in self.F])
            return g, ep
        return None
