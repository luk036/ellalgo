from typing import List, Optional, Tuple

import numpy as np

from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMI0Oracle:
    """
    Oracle for the Linear Matrix Inequality (LMI) constraint: F(x) ⪰ 0.

    This class is a specialized oracle for solving the LMI feasibility problem
    where the constant matrix `B` is zero. The constraint is of the form:

        F(x) = F₁x₁ + F₂x₂ + ... + Fₙxₙ ⪰ 0

    where `Fᵢ` are symmetric matrices and `x` is the vector of decision
    variables.

    The `assess_feas` method checks if a given solution `x` satisfies the LMI
    constraint. If it does, the method returns `None`. If not, it returns a
    separating hyperplane (a "cut") that separates the infeasible point from
    the feasible set.
    """

    def __init__(self, mat_f: List[np.ndarray]):
        """Initialize LMI oracle with coefficient matrices

        Args:
            mat_f (List[np.ndarray]): List of symmetric coefficient matrices [F₁, F₂, ..., Fₙ]
                Each F_k must be square matrix of same dimension
                mat_f[0] determines the matrix size m×m
        """
        self.mat_f = mat_f  # Store coefficient matrices
        # Initialize LDLT factorization manager with matrix dimension from F₁
        self.ldlt_mgr = LDLTMgr(len(mat_f[0]))

    def assess_feas(self, x: np.ndarray) -> Optional[Cut]:
        """
        Assess the feasibility of a candidate solution `x`.

        This method checks if the given solution `x` satisfies the LMI
        constraint `F(x) ⪰ 0`. It does this by constructing the matrix `F(x)`
        and performing an LDLT factorization to determine if it is positive
        semidefinite.

        Args:
            x (np.ndarray): The candidate solution vector.

        Returns:
            Optional[Cut]: `None` if `x` is feasible (i.e., the LMI constraint
            is satisfied). Otherwise, a tuple `(g, ep)` representing a
            separating hyperplane, where `g` is the subgradient and `ep` is
            the measure of violation.
        """

        def get_elem(i: int, j: int) -> float:
            """Construct element (i,j) of F(x) = ∑ x_k F_k

            Enables on-demand element computation without full matrix construction.
            This sparse approach saves memory for large-scale problems.
            """
            n = len(x)
            return sum(self.mat_f[k][i, j] * x[k] for k in range(n))

        # Attempt LDLT factorization (fails if matrix not PSD)
        if not self.ldlt_mgr.factor(get_elem):
            # Compute infeasibility certificate
            ep = self.ldlt_mgr.witness()  # Witness vector v such that vᵀF(x)v < 0
            # Calculate subgradient components: g_k = -vᵀF_k v
            g = np.array([-self.ldlt_mgr.sym_quad(Fk) for Fk in self.mat_f])
            return g, ep
        return None
