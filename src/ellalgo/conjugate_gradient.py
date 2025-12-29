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
    dimension = len(b)
    if x0 is None:
        solution = np.zeros(
            dimension
        )  # Initialize solution vector with zeros if no initial guess
    else:
        solution = x0.copy()  # Use provided initial guess

    # Initial residual calculation: residual = b - A*solution
    residual = b - np.dot(A, solution)
    direction = residual.copy()  # Initial search direction is set to residual
    residual_norm_sq = np.dot(residual, residual)  # Squared norm of residual

    for iteration in range(max_iter):
        A_direction = np.dot(A, direction)  # Matrix-vector product for line search
        step_size = residual_norm_sq / np.dot(
            direction, A_direction
        )  # Step size calculation
        solution += step_size * direction  # Update solution vector
        residual -= step_size * A_direction  # Update residual
        residual_norm_sq_new = np.dot(residual, residual)  # New residual norm squared

        # Check convergence condition using residual norm
        if np.sqrt(residual_norm_sq_new) < tol:
            return solution

        improvement_ratio = (
            residual_norm_sq_new / residual_norm_sq
        )  # Calculate improvement ratio
        direction = (
            residual + improvement_ratio * direction
        )  # Update search direction using conjugate gradient
        residual_norm_sq = (
            residual_norm_sq_new  # Update residual norm for next iteration
        )

    raise ValueError(f"Conj Grad did not converge after {max_iter} iterations")
