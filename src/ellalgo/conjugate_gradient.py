# conjugate_gradient.py

import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-5, max_iter=1000):
    """
    Solve Ax = b using the Conjugate Gradient method.

    Parameters:
    A : 2D numpy array
        The coefficient matrix (must be symmetric and positive definite)
    b : 1D numpy array
        The right-hand side vector
    x0 : 1D numpy array, optional
        Initial guess for the solution (default is zero vector)
    tol : float, optional
        Tolerance for convergence (default is 1e-5)
    max_iter : int, optional
        Maximum number of iterations (default is 1000)

    Returns:
    x : 1D numpy array
        The solution vector
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - np.dot(A, x)
    p = r.copy()
    r_norm_sq = np.dot(r, r)

    for i in range(max_iter):
        Ap = np.dot(A, p)
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = np.dot(r, r)

        if np.sqrt(r_norm_sq_new) < tol:
            return x

        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new

    raise ValueError(f"Conj Grad did not converge after {max_iter} iterations")