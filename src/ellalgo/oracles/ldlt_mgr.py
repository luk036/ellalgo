# -*- coding: utf-8 -*-
import math
from typing import Callable
import numpy as np


class LDLTMgr:
    """LDLT factorization (mainly for LMI oracles)

    `LDLTMgr` is a class that performs the LDLT factorization for a given
    symmetric matrix. The LDLT factorization decomposes a symmetric matrix A into
    the product of a lower triangular matrix L, a diagonal matrix D, and the
    transpose of L. This factorization is useful for solving linear systems and
    eigenvalue problems. The class provides methods to perform the factorization,
    check if the matrix is positive definite, calculate a witness vector if it is
    not positive definite, and calculate the symmetric quadratic form.

    - LDL^T square-root-free version
    - Option allow semidefinite
    - Cholesky–Banachiewicz style, row-based
    - Lazy evaluation
    - A matrix A in R^{m x m} is positive definite
                         iff v' A v > 0 for all v in R^n.
    - O(p^3) per iteration, independent of N
    """

    __slots__ = ("pos", "v", "_ndim", "_Temp", "allow_semidefinite")

    def __init__(self, N: int):
        """
        The above function is the constructor for a LDLT Ext object, which initializes various attributes
        and pre-allocates storage.

        :param N: The parameter N represents the dimension of the object. It is an integer value that
        determines the size of the object being constructed
        :type N: int
        """
        self.pos = (0, 0)
        self.v: np.ndarray = np.zeros(N)

        self._ndim: int = N
        self._Temp: np.ndarray = np.zeros((N, N))  # pre-allocate storage

    def factorize(self, A: np.ndarray) -> bool:
        """
        The function factorize performs LDLT Factorization on a symmetric matrix A and returns a boolean
        value indicating whether the factorization was successful.

        If $A$ is positive definite, then $p$ is zero.
        If it is not, then $p$ is a positive integer,
        such that $v = R^{-1} e_p$ is a certificate vector
        to make $v'*A[:p,:p]*v < 0$

        :param A: A is a numpy array representing a symmetric matrix
        :type A: np.ndarray
        :return: the result of calling the `factor` method with a lambda function as an argument.
        """
        return self.factor(lambda i, j: A[i, j])

    def factor(self, get_elem: Callable[[int, int], float]) -> bool:
        """
        The function performs LDLT Factorization on a symmetric matrix using lazy evaluation.

        :param get_elem: The `get_elem` parameter is a callable function that is used to access the elements
        of a symmetric matrix. It takes two integer arguments `i` and `j` and returns the value of the
        element at the `(i, j)` position in the matrix
        :type get_elem: Callable[[int, int], float]
        :return: The function `factor` returns a boolean value indicating whether the matrix is symmetric
        positive definite (SPD).
        """
        start = 0  # range start
        self.pos = (0, 0)
        for i in range(self._ndim):
            # j = start
            d = get_elem(i, start)
            for j in range(start, i):
                self._Temp[j, i] = d  # keep it for later use
                self._Temp[i, j] = d / self._Temp[j, j]  # the L[i, j]
                s = j + 1
                d = get_elem(i, s) - self._Temp[i, start:s].dot(self._Temp[start:s, s])
            self._Temp[i, i] = d
            if d <= 0.0:
                self.pos = start, i + 1
                break
        return self.is_spd()

    def factor_with_allow_semidefinite(
        self, get_elem: Callable[[int, int], float]
    ) -> bool:
        """
        The function performs LDLT Factorization on a symmetric matrix using lazy evaluation and checks
        if the matrix is positive definite.

        :param get_elem: The `get_elem` parameter is a callable function that takes two integer arguments
        `i` and `j` and returns a float value. This function is used to access the elements of a symmetric
        matrix `A`. The `factor_with_allow_semidefinite` method performs LDLT Factorization on
        :type get_elem: Callable[[int, int], float]
        :return: The function `factor_with_allow_semidefinite` returns a boolean value indicating whether
        the matrix is symmetric positive definite (SPD).
        """
        start = 0  # range start
        self.pos = (0, 0)
        for i in range(self._ndim):
            # j = start
            d = get_elem(i, start)
            for j in range(start, i):
                self._Temp[j, i] = d  # keep it for later use
                self._Temp[i, j] = d / self._Temp[j, j]  # the L[i, j]
                s = j + 1
                d = get_elem(i, s) - self._Temp[i, start:s].dot(self._Temp[start:s, s])
            self._Temp[i, i] = d
            if d < 0.0:
                self.pos = start, i + 1
                break
            elif d == 0:
                start = i + 1  # T[i, i] == 0 (very unlikely), restart at i+1
        return self.is_spd()

    def is_spd(self):
        """
        The function `is_spd` checks if a matrix `A` is symmetric positive definite (spd) and returns `True`
        if it is.
        :return: a boolean value. It returns True if the matrix A is symmetric positive definite (spd), and
        False otherwise.
        """
        return self.pos[1] == 0

    def witness(self) -> float:
        """
        The function "witness" provides evidence that a matrix is not symmetric positive definite.
            (square-root-free version)

           evidence: v' A v = -ep

        Raises:
            AssertionError: $A$ indeeds a spd matrix

        Returns:
            float: ep
        """
        if self.is_spd():
            raise AssertionError()
        start, n = self.pos
        m = n - 1
        self.v[m] = 1.0
        for i in range(m, start, -1):
            self.v[i - 1] = -self._Temp[i:n, i - 1].dot(self.v[i:n])
        return -self._Temp[m, m]

    def sym_quad(self, A: np.ndarray):
        """
        The `sym_quad` function calculates the quadratic form of a vector `v` with a symmetric matrix `A`.

        :param A: A is a numpy array
        :type A: np.ndarray
        :return: The function `sym_quad` returns the result of the dot product between `v` and the matrix
        product of `A[s:n, s:n]` and `v`.
        """
        s, n = self.pos
        v = self.v[s:n]
        return v.dot(A[s:n, s:n] @ v)

    def sqrt(self) -> np.ndarray:
        """Return upper triangular matrix R where A = R' * R

        Raises:
            AssertionError: [description]

        Returns:
            np.ndarray: [description]
        """
        if not self.is_spd():
            raise AssertionError()
        M = np.zeros((self._ndim, self._ndim))
        for i in range(self._ndim):
            M[i, i] = math.sqrt(self._Temp[i, i])
            for j in range(i + 1, self._ndim):
                M[i, j] = self._Temp[j, i] * M[i, i]
        return M


def test_ldlt_mgr_sqrt():
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    ldlt_obj = LDLTMgr(3)
    ldlt_obj.factor(lambda i, j: A[i, j])
    R = ldlt_obj.sqrt()
    assert np.allclose(R, np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]]))


if __name__ == "__main__":
    pass
