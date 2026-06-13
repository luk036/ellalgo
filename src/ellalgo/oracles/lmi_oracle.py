"""
LMI (Linear Matrix Inequality) feasibility oracle.

The `LMIOracle` class implements a feasibility oracle for LMI constraints
of the form:

    B - (F₁x₁ + F₂x₂ + ... + Fₙxₙ) ⪰ 0

where B and Fᵢ are symmetric matrices. It uses lazy element-wise matrix
construction and LDL^T factorization to check positive semidefiniteness,
avoiding construction of the full matrix whenever possible.

When a point is infeasible, the oracle returns a separating hyperplane (cut)
derived from the LDL^T witness vector, enabling the cutting-plane algorithm
to narrow the search space.
"""

from typing import List, Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleFeas
from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMIOracle(OracleFeas):
    """
    Oracle for Linear Matrix Inequality (LMI) constraints.

    This class implements the `OracleFeas` interface for solving semidefinite
    feasibility problems involving Linear Matrix Inequalities (LMIs). An LMI
    constraint is of the form:

        B - (F₁x₁ + F₂x₂ + ... + Fₙxₙ) ⪰ 0

    where `B` and `Fᵢ` are symmetric matrices, and `x` is the vector of decision
    variables. The notation `⪰ 0` means that the resulting matrix is required to
    be positive semidefinite.

    The `assess_feas` method checks if a given solution `x` satisfies the LMI
    constraint. If it does, the method returns `None`. If not, it returns a
    separating hyperplane (a "cut") that separates the infeasible point from
    the feasible set.
    """

    def __init__(self, mat_f: List[np.ndarray], mat_b: np.ndarray):
        """Initialize LMI Oracle with problem matrices.

        The constructor sets up the LMI constraint structure:
        (B - F₁x₁ - F₂x₂ - ... - Fₙxₙ) ⪰ 0

        :param mat_f: List of coefficient matrices [F₁, F₂, ..., Fₙ] where each F_i ∈ ℝ^{m×m}
        :param mat_b: Constant matrix B ∈ ℝ^{m×m} defining the LMI constraint
        """
        self.mat_f = mat_f  # Coefficient matrices for variables
        self.mat_f0 = mat_b  # Constant term matrix in LMI
        self.ldlt_mgr = LDLTMgr(
            len(mat_b)
        )  # Factorization manager for LDLT decomposition

    def assess_feas(self, xc: np.ndarray) -> Optional[Cut]:
        """
        Assess the feasibility of a candidate solution `xc`.

        This method checks if the given solution `xc` satisfies the LMI
        constraint. It does this by constructing the matrix `M(xc)` and
        performing an LDLT factorization to determine if it is positive
        semidefinite.

        Args:
            xc (np.ndarray): The candidate solution vector.

        Returns:
            Optional[Cut]: `None` if `xc` is feasible (i.e., the LMI constraint
            is satisfied). Otherwise, a tuple `(g, ep)` representing a
            separating hyperplane, where `g` is the subgradient and `ep` is
            the measure of violation.
        """

        def get_elem(i: int, j: int) -> float:
            """Construct element (i,j) of M(xc) = B - ∑ F_k*xc_k.

            Implements the LMI matrix construction element-wise for factorization.
            This avoids full matrix construction, enabling sparse computation.
            """
            s = sum(Fk[i, j] * xk for Fk, xk in zip(self.mat_f, xc))
            return self.mat_f0[i, j] - s

        if self.ldlt_mgr.factor(get_elem):
            return None  # Matrix is PSD => feasible solution

        # If infeasible, compute cut information:
        ep = self.ldlt_mgr.witness()  # Witness vector for negative eigenvalue
        # Compute subgradient components through symmetric quadratic form
        g = np.array([self.ldlt_mgr.sym_quad(Fk) for Fk in self.mat_f])
        return g, ep
