"""
LMIOracle

This code defines a class called LMIOracle, which is designed to solve a specific type of mathematical problem known as a Linear Matrix Inequality (LMI) constraint. The purpose of this code is to determine if there exists a solution that satisfies the given constraint, and if not, to provide information about why it's not feasible.

The LMIOracle class takes two inputs when it's created: mat_f and mat_b. These are matrices that define the LMI constraint. mat_f is a list of numpy arrays, and mat_b is a single numpy array. These matrices represent the mathematical relationship that needs to be satisfied.

The main function in this class is called assess_feas, which takes a numpy array xc as input. This function tries to determine if xc is a feasible solution to the LMI constraint. If it is feasible, the function returns None. If it's not feasible, it returns what's called a "cut" - a tuple containing information about why the solution isn't feasible.

To achieve its purpose, the code uses a technique called LDLT factorization. This is a way of breaking down a matrix into simpler parts, which can help determine if the matrix satisfies certain properties. The LDLTMgr class (which is used but not defined in this code snippet) handles this factorization.

The assess_feas function works by first creating a new matrix using the input xc and the original matrices mat_f and mat_b. It then tries to perform the LDLT factorization on this new matrix. If the factorization is successful, it means xc is a feasible solution, and the function returns None. If the factorization fails, it means xc is not feasible, and the function calculates and returns the "cut" information.

An important part of the logic is the get_elem function inside assess_feas. This function calculates each element of the new matrix based on the original matrices and the input xc. This is where the main mathematical operation of the LMI constraint is performed.

In summary, this code provides a way to check if a given solution satisfies a complex mathematical constraint, and if not, it provides information about why the solution doesn't work. This could be useful in optimization problems where you're trying to find a solution that satisfies certain mathematical conditions.
"""

from typing import Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleFeas
from ellalgo.oracles.ldlt_mgr import LDLTMgr

Cut = Tuple[np.ndarray, float]


class LMIOracle(OracleFeas):
    """Oracle for Linear Matrix Inequality constraint.

    This oracle solves the following feasibility problem:

    |    find  x
    |    s.t.  (B − F * x) ⪰ 0

    """

    def __init__(self, mat_f, mat_b):
        """
        Initializes a new LMIOracle object with the given matrix arguments.

        :param mat_f: A list of numpy arrays representing the matrix F.
        :param mat_b: A numpy array representing the matrix B.
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
