# test_conjugate_gradient.py

import numpy as np
from ellalgo.conjugate_gradient import conjugate_gradient


def test_conjugate_gradient_simple():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_expected = np.array([0.0909091, 0.6363636])

    x = conjugate_gradient(A, b)

    assert np.allclose(x, x_expected, rtol=1e-5)


# def test_conjugate_gradient_larger():
#     n = 100
#     A = np.diag(np.arange(1, n + 1))
#     x_true = np.random.rand(n)
#     b = np.dot(A, x_true)
#
#     x = conjugate_gradient(A, b)
#
#     assert np.allclose(x, x_true, rtol=1e-5)


def test_conjugate_gradient_with_initial_guess():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x0 = np.array([1.0, 1.0])
    x_expected = np.array([0.0909091, 0.6363636])

    x = conjugate_gradient(A, b, x0=x0)

    assert np.allclose(x, x_expected, rtol=1e-5)


# def test_conjugate_gradient_non_convergence():
#     A = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite
#     b = np.array([1.0, 1.0])
#
#     with pytest.raises(ValueError):
#         conjugate_gradient(A, b, max_iter=10)


def test_conjugate_gradient_tolerance():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    tol = 1e-10

    x = conjugate_gradient(A, b, tol=tol)

    residual = np.linalg.norm(b - np.dot(A, x))
    assert residual < tol
