from typing import Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleFeas
from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMIOldOracle(OracleFeas):
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, mat_f, mat_b):
        """
        The function initializes the class with two matrices and creates an instance of the LDLTMgr class.

        :param mat_f: A list of numpy arrays representing the matrix F
        :param mat_b: A numpy array representing the matrix B
        """
        self.mat_f = mat_f
        self.mat_f0 = mat_b
        self.ldlt_mgr = LDLTMgr(len(mat_b))

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """
        The `assess_feas` function assesses the feasibility of a given input array `x` and returns a `Cut`
        object if the feasibility is violated, otherwise it returns `None`.

        :param x: An array of values that will be used in the calculation
        :type x: np.ndarray
        :return: The function `assess_feas` returns an `Optional[Cut]`.
        """
        n = len(x)
        A = self.mat_f0.copy()
        A -= sum(self.mat_f[k] * x[k] for k in range(n))
        if not self.ldlt_mgr.factorize(A):
            ep = self.ldlt_mgr.witness()
            g = np.array([self.ldlt_mgr.sym_quad(self.mat_f[i]) for i in range(n)])
            return g, ep
        return None
