from typing import Optional, Tuple

import numpy as np

from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMI0Oracle:
    """Oracle for Linear Matrix Inequality (LMI) constraint: F(x) ⪰ 0

    Solves the feasibility problem:
        Find x ∈ ℝⁿ such that ∑_{k=1}^n F_k x_k ≽ 0
    Where:
        - F_k ∈ 𝕊^m (symmetric matrices) are given in mat_f
        - x = [x_1, ..., x_n]^T is the decision vector
        - ≽ denotes positive semidefinite (PSD) constraint

    The oracle uses LDLT factorization to verify PSD property
    and generates cutting planes for infeasible solutions.
    """

    def __init__(self, mat_f):
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
        """Assess feasibility of solution x against LMI constraint

        Implementation Strategy:
        1. Construct matrix F(x) = ∑ x_k F_k through element-wise computation
        2. Attempt LDLT factorization:
           - Success: F(x) is PSD (feasible) → return None
           - Failure: F(x) not PSD → compute cutting plane (g, σ)

        Args:
            x (np.ndarray): Candidate solution vector [x₁, ..., xₙ]^T ∈ ℝⁿ

        Returns:
            Optional[Cut]:
                - None if x is feasible (F(x) ≽ 0)
                - Tuple (g, σ) representing cutting plane gᵀ(y - x) ≥ σ otherwise
        """

        def get_elem(i, j):
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
