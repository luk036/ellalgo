# test_conjugate_gradient.py

import numpy as np

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


# def test_conjugate_gradient_non_convergence() -> None:
#     A = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite
#     b = np.array([1.0, 1.0])
#
#     with pytest.raises(ValueError):
#         conjugate_gradient(A, b, max_iter=10)


def test_conjugate_gradient_tolerance() -> None:
    matrix_A = np.array([[4.0, 1.0], [1.0, 3.0]])
    vector_b = np.array([1.0, 2.0])
    tolerance = 1e-10

    solution = conjugate_gradient(matrix_A, vector_b, tol=tolerance)

    residual = np.linalg.norm(vector_b - np.dot(matrix_A, solution))
    assert residual < tolerance
