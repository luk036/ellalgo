"""
Legacy LMI oracle with explicit matrix construction.

This module provides a feasibility oracle for Linear Matrix Inequality (LMI)
constraints. Unlike LMIOracle (which uses lazy element-wise evaluation), this
implementation constructs the full LMI matrix explicitly before factorization.
"""

from typing import List, Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleFeas
from ellalgo.oracles.ldlt_mgr import LDLTMgr
from ellalgo.oracles.lmi_oracle_base import LMIBase

Cut = Tuple[np.ndarray, float]


class LMIOldOracle(LMIBase, OracleFeas):
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

        find  x
        s.t.  (B − F * x) ⪰ 0

    This is a legacy implementation that constructs the full LMI matrix explicitly.
    For better performance with large matrices, use `LMIOracle` which uses lazy
    evaluation.

    Concrete LMI oracle: builds the matrix ``A = B − Σ_k F_k x_k`` eagerly and
    feeds a lambda element accessor with a positive ``sym_quad`` sign to the
    shared LMIBase skeleton. Behaviorally identical to `LMIOracle`, which uses
    lazy evaluation instead of a pre-built matrix.

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

    mat_f: List[np.ndarray]
    mat_f0: np.ndarray
    ldlt_mgr: LDLTMgr

    def __init__(self, mat_f: List[np.ndarray], mat_b: np.ndarray):
        """Initialize the LMI oracle with coefficient matrices.

        Args:
            mat_f: List of coefficient matrices [F₁, F₂, ..., Fₙ] where each F_i ∈ ℝ^{m×m}.
            mat_b: Constant matrix B ∈ ℝ^{m×m} defining the LMI constraint.
        """
        self.mat_f = mat_f
        self.mat_f0 = mat_b
        self.ldlt_mgr = LDLTMgr(len(mat_b))

    def assess_feas(self, xc: np.ndarray) -> Optional[Cut]:
        """Assess the feasibility of a candidate solution.

        This method checks if the given solution satisfies the LMI constraint
        (B − F₁x₁ − F₂x₂ − ... − Fₙxₙ) ⪰ 0 by constructing the full matrix
        and performing LDLT factorization.

        Args:
            xc: The candidate solution vector x.

        Returns:
            `None` if feasible, otherwise a tuple `(g, ep)` containing the
            subgradient `g` and the negative eigenvalue measure `ep`.

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
        return self._assess(lambda i, j: A[i, j], +1)
