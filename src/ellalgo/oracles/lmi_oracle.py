"""
LMIOracle

This code defines a class called LMIOracle, which is designed to solve a
specific type of mathematical problem known as a Linear Matrix Inequality (LMI)
constraint. The purpose of this code is to determine if there exists a solution
that satisfies the given constraint, and if not, to provide information about
why it's not feasible.

The LMIOracle class takes two inputs when it's created: mat_f and mat_b. These
are matrices that define the LMI constraint. mat_f is a list of numpy arrays,
and mat_b is a single numpy array. These matrices represent the mathematical
relationship that needs to be satisfied.

The main function in this class is called assess_feas, which takes a numpy
array xc as input. This function tries to determine if xc is a feasible
solution to the LMI constraint. If it is feasible, the function returns None.
If it's not feasible, it returns what's called a "cut" - a tuple containing
information about why the solution isn't feasible.

To achieve its purpose, the code uses a technique called LDLT factorization.
This is a way of breaking down a matrix into simpler parts, which can help
determine if the matrix satisfies certain properties. The LDLTMgr class (which
is used but not defined in this code snippet) handles this factorization.

The assess_feas function works by first creating a new matrix using the input
xc and the original matrices mat_f and mat_b. It then tries to perform the LDLT
factorization on this new matrix. If the factorization is successful, it means
xc is a feasible solution, and the function returns None. If the factorization
fails, it means xc is not feasible, and the function calculates and returns the
"cut" information.

An important part of the logic is the get_elem function inside assess_feas.
This function calculates each element of the new matrix based on the original
matrices and the input xc. This is where the main mathematical operation of the
LMI constraint is performed.

In summary, this code provides a way to check if a given solution satisfies a
complex mathematical constraint, and if not, it provides information about why
the solution doesn't work. This could be useful in optimization problems where
you're trying to find a solution that satisfies certain mathematical conditions.
"""

from typing import Optional, Tuple

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

    def __init__(self, mat_f, mat_b):
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

        def get_elem(i, j):
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
