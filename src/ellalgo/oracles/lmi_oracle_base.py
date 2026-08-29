"""
Shared skeleton for LMI feasibility oracles (Template Method).

The three LMI oracle flavors (LMIOracle, LMI0Oracle, LMIOldOracle) differ only
in how the matrix elements ``A(i,j)`` are assembled (lazy element access vs.
an eagerly-built matrix), in the ``sym_quad`` sign convention, and in the
presence of a constant term ``B``. The factorization / witness / cut-packing
pipeline is identical and lives here.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np

from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMIBase:
    """Shared skeleton for Linear Matrix Inequality feasibility oracles.

    Concrete LMI oracles build a lazy element accessor ``get_elem`` for the
    matrix ``A(x)`` and delegate to :meth:`_assess`, which runs the fixed
    pipeline: factor -> witness -> sym_quad -> pack cut. This is the Template
    Method pattern: the algorithm skeleton is fixed here; each concrete oracle
    supplies the matrix-construction strategy via ``get_elem`` and the sign.
    """

    mat_f: List[np.ndarray]  # Coefficient matrices for variables
    ldlt_mgr: LDLTMgr  # Factorization manager for LDLT decomposition

    def _assess(
        self, get_elem: Callable[[int, int], float], sign: int
    ) -> Optional[Cut]:
        """Shared assess_feas skeleton: factor, witness, sym_quad, pack cut.

        Args:
            get_elem: Callable returning matrix element A(i, j) on demand.
            sign: +1 or -1 convention for the sym_quad subgradient.

        Returns:
            None if A(x) is positive definite (feasible); otherwise a cut
            tuple (g, ep) where g is the subgradient and ep the violation.
        """
        if self.ldlt_mgr.factor(get_elem):
            return None  # Matrix is PSD => feasible solution
        # If infeasible, compute cut information:
        ep = self.ldlt_mgr.witness()  # Witness vector for negative eigenvalue
        # Compute subgradient components through symmetric quadratic form
        g = np.array([sign * self.ldlt_mgr.sym_quad(Fk) for Fk in self.mat_f])
        return g, ep
