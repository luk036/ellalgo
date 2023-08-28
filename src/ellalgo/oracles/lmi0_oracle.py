from typing import Optional, Tuple

import numpy as np

from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMI0Oracle:
    """Oracle for Linear Matrix Inequality constraint

    find  x
    s.t.  F * x ⪰ 0

    """

    def __init__(self, mat_f):
        """[summary]

        Arguments:
            mat_f (List[np.ndarray]): [description]
        """
        self.mat_f = mat_f
        self.ldlt_mgr = LDLTMgr(len(mat_f[0]))

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (np.ndarray): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def get_elem(i, j):
            n = len(x)
            return sum(self.mat_f[k][i, j] * x[k] for k in range(n))

        if not self.ldlt_mgr.factor(get_elem):
            ep = self.ldlt_mgr.witness()
            g = np.array([-self.ldlt_mgr.sym_quad(Fk) for Fk in self.mat_f])
            return g, ep
        return None
