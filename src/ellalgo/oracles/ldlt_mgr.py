# -*- coding: utf-8 -*-
"""
LDLTMgr (LDLT Manager)

This code defines a class called LDLTMgr, which stands for LDLT Manager. The purpose of this class is to perform a special kind of matrix factorization called LDLT factorization on symmetric matrices. This factorization is useful in various mathematical and engineering applications, especially when dealing with linear systems and eigenvalue problems.

The main input for this class is a symmetric matrix, which can be provided either as a NumPy array or through a function that returns individual matrix elements. The primary output is a boolean value indicating whether the input matrix is positive definite (a special property of matrices). Additionally, the class can produce other outputs like a witness vector (if the matrix is not positive definite) and a square root of the matrix (if it is positive definite).

The LDLTMgr class achieves its purpose through several methods. The main method is 'factor', which performs the actual LDLT factorization. It does this by going through the matrix elements row by row, calculating and storing values in a special way. This process helps determine if the matrix is positive definite and allows for efficient calculations later.

An important aspect of this code is its use of "lazy evaluation". This means it doesn't need the entire matrix upfront but can work with just a function that provides matrix elements as needed. This can be more efficient for large matrices or when the matrix is defined by a formula rather than stored values.

The class also includes methods to check if the matrix is positive definite ('is_spd'), calculate a witness vector if it's not ('witness'), and compute a symmetric quadratic form ('sym_quad'). These additional functionalities make the class versatile for various matrix-related tasks.

One of the key transformations happening in this code is the factorization itself. It's breaking down the original matrix into simpler parts (L, D, and L transpose), which can be used to solve problems more easily. This factorization is done in a way that avoids using square roots, which can be computationally expensive.

Overall, the LDLTMgr class provides a set of tools for working with symmetric matrices, with a focus on determining their properties (like positive definiteness) and performing useful calculations efficiently. It's designed to be flexible and can handle both standard matrices and those defined by functions, making it useful in a variety of mathematical and engineering contexts.
"""

import math
from typing import Callable, Tuple

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
    - Cholesky-Banachiewicz style, row-based
    - Lazy evaluation
    - A matrix A in R^{m x m} is positive definite
                         iff v^T A v > 0 for all v in R^n.
    - O(p^3) per iteration, independent of ndim

    Examples:
        >>> A = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
        >>> ldl = LDLTMgr(3)
        >>> ldl.factorize(A)
        True
    """

    __slots__ = ("pos", "wit", "_ndim", "_storage")

    def __init__(self, ndim: int):
        """
        Initializes the LDLT manager with given matrix dimension.

        Args:
            ndim: The dimension of the square matrix to be factorized.

        Attributes initialized:
            pos: Tuple tracking the position where positive definiteness fails (0,0) initially
            wit: Witness vector storage initialized to zeros
            _ndim: Stores the matrix dimension
            _storage: Pre-allocated storage for factorization results (ndim x ndim matrix)

        The initialization prepares all necessary storage to avoid repeated allocations during factorization.
        """
        self.pos: Tuple[int, int] = (0, 0)
        self.wit: np.ndarray = np.zeros(ndim)
        self._ndim: int = ndim
        self._storage: np.ndarray = np.zeros((ndim, ndim))  # pre-allocate storage

    def factorize(self, mat: np.ndarray) -> bool:
        """
        Performs LDLT factorization on a given numpy array matrix.

        This is a convenience wrapper around the factor() method that takes a matrix directly
        rather than an element-accessor function.

        Args:
            mat: A symmetric numpy array to be factorized

        Returns:
            bool: True if matrix is positive definite, False otherwise

        Examples:
            >>> mat = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factorize(mat)
            True
        """
        return self.factor(lambda i, j: mat[i, j])

    def factor(self, get_elem: Callable[[int, int], float]) -> bool:
        """
        Performs LDLT factorization using lazy element access.

        The factorization proceeds row by row, computing diagonal entries and off-diagonal
        multipliers. If any diagonal entry becomes non-positive, factorization stops early
        and records the failure position.

        Args:
            get_elem: Function that returns matrix element at (i,j) position

        Returns:
            bool: True if matrix is positive definite (all diagonal entries positive)

        The factorization stores results in _storage:
        - Diagonal entries (D matrix) are stored in _storage[i,i]
        - Off-diagonal entries (L matrix) are stored in _storage[i,j] for j < i

        Examples:
            >>> mat = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factor(lambda i, j: mat[i, j])
            True
        """
        start: int = 0  # range start
        self.pos = (0, 0)
        for i in range(self._ndim):
            diag = get_elem(i, start)
            for j in range(start, i):
                self._storage[j, i] = diag  # keep it for later use
                self._storage[i, j] = diag / self._storage[j, j]  # the L[i, j]
                stop = j + 1
                diag = get_elem(i, stop) - self._storage[i, start:stop].dot(
                    self._storage[start:stop, stop]
                )
            self._storage[i, i] = diag
            if diag <= 0.0:
                self.pos = start, i + 1
                break
        return self.is_spd()

    def factor_with_allow_semidefinite(
        self, get_elem: Callable[[int, int], float]
    ) -> bool:
        """
        Performs LDLT factorization allowing for positive semi-definite matrices.

        Similar to factor() but handles zero diagonal entries (indicating semi-definiteness)
        by restarting factorization from the next row.

        Args:
            get_elem: Function that returns matrix element at (i,j) position

        Returns:
            bool: True if matrix is positive semi-definite (no negative diagonal entries)

        This version is more tolerant of zero diagonal entries than factor(), which
        requires strictly positive entries for positive definiteness.

        Examples:
            >>> mat = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factor_with_allow_semidefinite(lambda i, j: mat[i, j])
            True
        """
        start: int = 0  # range start
        self.pos = (0, 0)
        for i in range(self._ndim):
            diag = get_elem(i, start)
            for j in range(start, i):
                self._storage[j, i] = diag  # keep it for later use
                self._storage[i, j] = diag / self._storage[j, j]  # the L[i, j]
                stop = j + 1
                diag = get_elem(i, stop) - self._storage[i, start:stop].dot(
                    self._storage[start:stop, stop]
                )
            self._storage[i, i] = diag
            if diag < 0.0:
                self.pos = start, i + 1
                break
            elif diag == 0:
                start = i + 1  # T[i, i] == 0 (very unlikely), restart at i+1
        return self.is_spd()

    def is_spd(self):
        """
        Checks if the matrix is symmetric positive definite (SPD).

        Returns:
            bool: True if the matrix is SPD (pos[1] == 0), False otherwise

        The check is based on whether any diagonal entry was non-positive during
        factorization, which would have set pos[1] to a non-zero value.

        Examples:
            >>> mat = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factorize(mat)
            True
            >>> ldl.is_spd()
            True
        """
        return self.pos[1] == 0

    def witness(self) -> float:
        """
        Computes a witness vector proving the matrix is not positive definite.

        Returns:
            float: The negative eigenvalue (ep) showing v^T A v = -ep < 0

        Raises:
            AssertionError: If called on a positive definite matrix

        The witness vector is stored in self.wit and can be accessed after calling
        this method. The vector satisfies v^T A v < 0 for the failed submatrix.

        Examples:
            >>> mat = np.array([[1.0, 2.0, 3.0], [2.0, 3.5, 5.0], [3.0, 5.0, 6.0]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factorize(mat)
            False
            >>> ldl.witness()
            0.5
        """
        if self.is_spd():
            raise AssertionError()
        start, pos = self.pos
        m = pos - 1
        self.wit[m] = 1.0
        for i in range(m, start, -1):
            self.wit[i - 1] = -self._storage[i:pos, i - 1].dot(self.wit[i:pos])
        return -self._storage[m, m]

    def sym_quad(self, mat: np.ndarray):
        """
        Computes the quadratic form v^T M v using the witness vector.

        Args:
            mat: The matrix M to compute the quadratic form with

        Returns:
            float: The value of v^T M v where v is the witness vector

        Note: witness() must be called first to set up the witness vector.
        The computation uses only the submatrix where positive definiteness failed.

        Examples:
            >>> mat = np.array([[1.0, 2.0, 3.0], [2.0, 3.5, 5.0], [3.0, 5.0, 6.0]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factorize(mat)
            False
            >>> ldl.pos
            (0, 2)
            >>> ldl.witness() # call this before sym_quad()
            0.5
            >>> ldl.wit
            array([-2.,  1.,  0.])
            >>> mat_b = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
            >>> ldl.sym_quad(mat_b)
            3.25
        """
        start, ndim = self.pos
        wit = self.wit[start:ndim]
        return wit.dot(mat[start:ndim, start:ndim] @ wit)

    def sqrt(self) -> np.ndarray:
        """
        Computes the upper triangular square root matrix R where A = R^T R.

        Returns:
            np.ndarray: Upper triangular matrix R

        Raises:
            AssertionError: If matrix is not positive definite

        This is essentially the Cholesky decomposition, computed from the LDLT
        factors without directly computing square roots until the final step.

        Examples:
            >>> mat = np.array([[1.0, 0.5, 0.5], [0.5, 1.25, 0.75], [0.5, 0.75, 1.5]])
            >>> ldl = LDLTMgr(3)
            >>> ldl.factorize(mat)
            True
            >>> ldl.sqrt()
            array([[1. , 0.5, 0.5],
                   [0. , 1. , 0.5],
                   [0. , 0. , 1. ]])
        """
        if not self.is_spd():
            raise AssertionError()
        R = np.zeros((self._ndim, self._ndim))
        for i in range(self._ndim):
            R[i, i] = math.sqrt(self._storage[i, i])
            for j in range(i + 1, self._ndim):
                R[i, j] = self._storage[j, i] * R[i, i]
        return R
