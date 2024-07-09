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

    def __init__(self, mat_f, mat_b):
        """
        The function initializes a new LMI oracle object with given arguments.

        :param mat_f: A list of numpy arrays.
        :param mat_b: mat_b is a numpy array
        """
        self.mat_f = mat_f
        self.mat_f0 = mat_b
        self.ldlt_mgr = LDLTMgr(len(mat_b))

    def assess_feas(self, xc: np.ndarray) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input and returns a cut if it is not
        feasible.

        :param x: An input array of type `np.ndarray`
        :type x: np.ndarray
        :return: The function `assess_feas` returns an optional `Cut` object.
        """

        def get_elem(i, j):
            return self.mat_f0[i, j] - sum(
                Fk[i, j] * xk for Fk, xk in zip(self.mat_f, xc)
            )

        if self.ldlt_mgr.factor(get_elem):
            return None
        ep = self.ldlt_mgr.witness()
        g = np.array([self.ldlt_mgr.sym_quad(Fk) for Fk in self.mat_f])
        return g, ep
