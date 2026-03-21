from typing import List, Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleFeas
from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMIOldOracle(OracleFeas):
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    This is a legacy implementation that constructs the full LMI matrix explicitly.
    For better performance with large matrices, use `LMIOracle` which uses lazy
    evaluation.

    Examples:
        >>> import numpy as np
        >>> from ellalgo.oracles.lmi_old_oracle import LMIOldOracle
        >>> F1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> F2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        >>> B = np.array([[2.0, 0.0], [0.0, 2.0]])
        >>> oracle = LMIOldOracle([F1, F2], B)
        >>> result = oracle.assess_feas(np.array([0.0, 0.0]))
        >>> result is None or isinstance(result, tuple)
        True
    """

    def __init__(self, mat_f: List[np.ndarray], mat_b: np.ndarray):
        """Initialize the LMI oracle with coefficient matrices.

        :param mat_f: List of coefficient matrices [F₁, F₂, ..., Fₙ] where each F_i ∈ ℝ^{m×m}
        :param mat_b: Constant matrix B ∈ ℝ^{m×m} defining the LMI constraint
        """
        self.mat_f = mat_f
        self.mat_f0 = mat_b
        self.ldlt_mgr = LDLTMgr(len(mat_b))

    def assess_feas(self, xc: np.ndarray) -> Optional[Cut]:
        """Assess the feasibility of a candidate solution.

        This method checks if the given solution satisfies the LMI constraint
        (B − F₁x₁ − F₂x₂ − ... − Fₙxₙ) ⪰ 0 by constructing the full matrix
        and performing LDLT factorization.

        :param xc: The candidate solution vector x
        :returns: `None` if feasible, otherwise a tuple `(g, ep)` containing the
            subgradient `g` and the negative eigenvalue measure `ep`
        :raises: None

        Examples:
            >>> import numpy as np
            >>> from ellalgo.oracles.lmi_old_oracle import LMIOldOracle
            >>> F1 = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> F2 = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> B = np.array([[2.0, 0.0], [0.0, 2.0]])
            >>> oracle = LMIOldOracle([F1, F2], B)
            >>> oracle.assess_feas(np.array([0.0, 0.0])) is None
            True
        """
        n = len(xc)
        A = self.mat_f0.copy()
        A -= sum(self.mat_f[k] * xc[k] for k in range(n))
        if not self.ldlt_mgr.factorize(A):
            ep = self.ldlt_mgr.witness()
            g = np.array([self.ldlt_mgr.sym_quad(self.mat_f[i]) for i in range(n)])
            return g, ep
        return None
