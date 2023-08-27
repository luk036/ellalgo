from typing import Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleFeas
from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMIOracle(OracleFeas):
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, F, B):
        """
        The function initializes a new lmi oracle object with given arguments.
        
        :param F: A list of numpy arrays. It is not clear what these arrays represent without further
        context
        :param B: B is a numpy array
        """
        self.F = F
        self.F0 = B
        self.Q = LDLTMgr(len(B))

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input and returns a cut if it is not
        feasible.
        
        :param x: An input array of type `np.ndarray`
        :type x: np.ndarray
        :return: The function `assess_feas` returns an optional `Cut` object.
        """

        def get_elem(i, j):
            return self.F0[i, j] - sum(Fk[i, j] * xk for Fk, xk in zip(self.F, x))

        if self.Q.factor(get_elem):
            return None
        ep = self.Q.witness()
        g = np.array([self.Q.sym_quad(Fk) for Fk in self.F])
        return g, ep
