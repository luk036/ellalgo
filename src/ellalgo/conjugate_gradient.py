# conjugate_gradient.py

import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.

    The Conjugate Gradient method is an iterative algorithm for solving symmetric positive definite linear systems. It is particularly efficient for large, sparse systems.

    Algorithm Steps:
    1. Initialize residual (r), search direction (p), and solution vector (x)
    2. Iteratively update solution using orthogonal search directions
    3. Maintain conjugacy of search directions to ensure convergence in at most n steps

    Args:
        A (numpy.ndarray): The coefficient matrix (must be symmetric and positive definite).
        b (numpy.ndarray): The right-hand side vector.
        x0 (numpy.ndarray, optional): Initial guess for the solution (default is zero vector).
        tol (float, optional): Tolerance for convergence (default is 1e-5).
        max_iter (int, optional): Maximum number of iterations (default is 1000).

    Returns:
        numpy.ndarray: The solution vector.

    Raises:
        ValueError: If the Conjugate Gradient method does not converge after the maximum number of iterations.
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
