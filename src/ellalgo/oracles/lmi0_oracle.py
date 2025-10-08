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
        """Assess the feasibility of a solution `x` against the LMI constraint.

        This method checks if the matrix `F(x)` is positive semidefinite (PSD).

        Implementation Strategy:
            1. Construct the matrix `F(x) = sum(x_k * F_k)` using an element-wise
               approach to save memory.
            2. Attempt to perform an LDLT factorization of `F(x)`.
            3. If the factorization is successful, it means `F(x)` is PSD, and the
               solution `x` is feasible. In this case, the method returns `None`.
            4. If the factorization fails, it means `F(x)` is not PSD. The method
               then computes a cutting plane `(g, sigma)` that separates `x` from
               the feasible region.

        Arguments:
            x (np.ndarray): The candidate solution vector.

        Returns:
            Optional[Cut]:
                - `None` if `x` is feasible (i.e., `F(x)` is PSD).
                - A tuple `(g, sigma)` representing the cutting plane if `x` is
                  infeasible.
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
