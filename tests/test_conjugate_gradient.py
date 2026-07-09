# test_conjugate_gradient.py

import numpy as np
import pytest

from ellalgo.conjugate_gradient import conjugate_gradient


def test_conjugate_gradient_simple() -> None:
    matrix_A = np.array([[4.0, 1.0], [1.0, 3.0]])
    vector_b = np.array([1.0, 2.0])
    expected_solution = np.array([0.0909091, 0.6363636])

    solution = conjugate_gradient(matrix_A, vector_b)

    assert np.allclose(solution, expected_solution, rtol=1e-5)


# def test_conjugate_gradient_larger() -> None:
#     n = 100
#     A = np.diag(np.arange(1, n + 1))
#     x_true = np.random.rand(n)
#     b = np.dot(A, x_true)
#
#     x = conjugate_gradient(A, b)
#
#     assert np.allclose(x, x_true, rtol=1e-5)


def test_conjugate_gradient_with_initial_guess() -> None:
    matrix_A = np.array([[4.0, 1.0], [1.0, 3.0]])
    vector_b = np.array([1.0, 2.0])
    initial_guess = np.array([1.0, 1.0])
    expected_solution = np.array([0.0909091, 0.6363636])

    solution = conjugate_gradient(matrix_A, vector_b, x0=initial_guess)

    assert np.allclose(solution, expected_solution, rtol=1e-5)


def test_conjugate_gradient_non_convergence() -> None:
    import pytest

    # Use a matrix that will cause division by zero in the algorithm
    A = np.array([[0.0, 0.0], [0.0, 0.0]])  # Zero matrix - will cause issues
    b = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="Conj Grad did not converge after"):
        conjugate_gradient(A, b, max_iter=10)


def test_conjugate_gradient_max_iter_exhausted() -> None:
    """Test CG raises ValueError when max_iter is exhausted without convergence (line 97).

    CG needs at most n iterations for an n×n system. Using max_iter=1 on a 2×2 SPD
    system forces the loop to finish without converging, hitting the post-loop raise.
    """
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Conj Grad did not converge after"):
        conjugate_gradient(A, b, max_iter=1, tol=1e-15)


def test_conjugate_gradient_tolerance() -> None:
    matrix_A = np.array([[4.0, 1.0], [1.0, 3.0]])
    vector_b = np.array([1.0, 2.0])
    tolerance = 1e-10

    solution = conjugate_gradient(matrix_A, vector_b, tol=tolerance)

    residual = np.linalg.norm(vector_b - np.dot(matrix_A, solution))
    assert residual < tolerance
