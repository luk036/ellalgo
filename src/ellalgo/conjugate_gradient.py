"""
Conjugate Gradient method for solving symmetric positive-definite linear systems.

This module implements the Conjugate Gradient (CG) iterative algorithm for
solving Ax = b where A is a symmetric positive-definite matrix. The CG method
generates A-orthogonal search directions and is guaranteed to converge in at
most n iterations in exact arithmetic.
"""

from typing import Optional

import numpy as np


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> np.ndarray:
    r"""Solve :math:`\mathbf{Ax} = \mathbf{b}` using the Conjugate Gradient method.

    The CG method generates :math:`\mathbf{A}`-orthogonal (conjugate) search
    directions and converges in at most :math:`n` iterations in exact arithmetic.
    The iteration proceeds as follows:

    .. math::

       \mathbf{r}_0 &= \mathbf{b} - \mathbf{A}\mathbf{x}_0,\qquad
       \mathbf{p}_0 = \mathbf{r}_0 \\[4pt]
       \alpha_k &= \frac{\mathbf{r}_k^T \mathbf{r}_k}
                         {\mathbf{p}_k^T \mathbf{A} \mathbf{p}_k} \\[4pt]
       \mathbf{x}_{k+1} &= \mathbf{x}_k + \alpha_k \mathbf{p}_k \\[4pt]
       \mathbf{r}_{k+1} &= \mathbf{r}_k - \alpha_k \mathbf{A} \mathbf{p}_k \\[4pt]
       \beta_k &= \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}
                       {\mathbf{r}_k^T \mathbf{r}_k} \\[4pt]
       \mathbf{p}_{k+1} &= \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k

    The algorithm terminates when :math:`\|\mathbf{r}_k\| < \text{tol}`.

    Args:
        A: Symmetric positive-definite coefficient matrix.
        b: Right-hand side vector.
        x0: Initial guess (default: zero vector).
        tol: Convergence tolerance on residual norm (default: 1e-5).
        max_iter: Maximum iterations (default: 1000).

    Returns:
        Solution vector :math:`\mathbf{x}` satisfying :math:`\mathbf{Ax} = \mathbf{b}`.

    Raises:
        ValueError: If the method does not converge within ``max_iter`` iterations.
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
        direction_dot_A_direction = np.dot(direction, A_direction)

        # Check for zero or near-zero denominator to avoid division by zero
        if direction_dot_A_direction == 0:
            raise ValueError(f"Conj Grad did not converge after {max_iter} iterations")

        step_size = (
            residual_norm_sq / direction_dot_A_direction
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
