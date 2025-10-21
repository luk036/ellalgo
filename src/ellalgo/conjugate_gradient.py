# conjugate_gradient.py

from typing import Optional

import numpy as np


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    Solves the linear system of equations Ax = b using the Conjugate Gradient method.

    The Conjugate Gradient method is an iterative algorithm designed for solving
    large, sparse linear systems where the matrix A is symmetric and
    positive-definite. It is one of the most popular and efficient methods for
    such problems.

    The key idea of the method is to generate a sequence of search directions
    that are A-orthogonal (or conjugate) to each other. This property ensures

    that the algorithm converges to the exact solution in at most n iterations,
    where n is the dimension of the system (assuming no rounding errors).

    Args:
        A (numpy.ndarray): The coefficient matrix of the linear system. It must be
            a symmetric and positive-definite matrix.
        b (numpy.ndarray): The right-hand side vector of the linear system.
        x0 (numpy.ndarray, optional): An initial guess for the solution. If not
            provided, a zero vector is used. Defaults to None.
        tol (float, optional): The tolerance for convergence. The iteration stops
            when the norm of the residual is less than this value. Defaults to 1e-5.
        max_iter (int, optional): The maximum number of iterations to perform.
            Defaults to 1000.

    Returns:
        numpy.ndarray: The solution vector x that satisfies Ax = b.

    Raises:
        ValueError: If the method does not converge within the specified maximum
            number of iterations.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)  # Initialize solution vector with zeros if no initial guess
    else:
        x = x0.copy()  # Use provided initial guess

    # Initial residual calculation: r = b - A*x
    r = b - np.dot(A, x)
    p = r.copy()  # Initial search direction is set to residual
    r_norm_sq = np.dot(r, r)  # Squared norm of residual

    for i in range(max_iter):
        Ap = np.dot(A, p)  # Matrix-vector product for line search
        alpha = r_norm_sq / np.dot(p, Ap)  # Step size calculation
        x += alpha * p  # Update solution vector
        r -= alpha * Ap  # Update residual
        r_norm_sq_new = np.dot(r, r)  # New residual norm squared

        # Check convergence condition using residual norm
        if np.sqrt(r_norm_sq_new) < tol:
            return x

        beta = r_norm_sq_new / r_norm_sq  # Calculate improvement ratio
        p = r + beta * p  # Update search direction using conjugate gradient
        r_norm_sq = r_norm_sq_new  # Update residual norm for next iteration

    raise ValueError(f"Conj Grad did not converge after {max_iter} iterations")
