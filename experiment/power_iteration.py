import numpy as np

"""
Power method for finding the largest eigenvalue of a square matrix
"""


class Options:
    def __init__(self, max_iters, tolerance):
        self.max_iters = max_iters
        self.tolerance = tolerance


def norm_l1(x):
    return np.sum(np.abs(x))


def power_iteration(A, x, options):
    """Power iteration method

    Performs the power iteration algorithm to find the largest eigenvalue and corresponding eigenvector of the input matrix A.

    Args:
        A (numpy.ndarray): The input square matrix.
        x (numpy.ndarray): The initial vector, assumed to be non-zero.
        options (Options): An Options object containing the maximum number of iterations and the tolerance for convergence.

    Returns:
        numpy.ndarray: The eigenvector corresponding to the largest eigenvalue.
        float: The largest eigenvalue.
        int: The number of iterations performed.
    """
    x /= np.sqrt(np.sum(x**2))
    for niter in range(options.max_iters):
        x1 = x
        x = A @ x1
        x /= np.sqrt(np.sum(x**2))
        if norm_l1(x - x1) <= options.tolerance or norm_l1(x + x1) <= options.tolerance:
            return x, x @ (A @ x), niter
    return x, x @ (A @ x), options.max_iters


def power_iteration4(A, x, options):
    """Power iteration method

    Performs the power iteration algorithm to find the largest eigenvalue and corresponding eigenvector of the input matrix A.

    Args:
        A (numpy.ndarray): The input square matrix.
        x (numpy.ndarray): The initial vector, assumed to be non-zero.
        options (Options): An Options object containing the maximum number of iterations and the tolerance for convergence.

    Returns:
        numpy.ndarray: The eigenvector corresponding to the largest eigenvalue.
        float: The largest eigenvalue.
        int: The number of iterations performed.
    """
    x /= norm_l1(x)
    for niter in range(options.max_iters):
        x1 = x
        x = A @ x1
        x /= norm_l1(x)
        if norm_l1(x - x1) <= options.tolerance or norm_l1(x + x1) <= options.tolerance:
            x /= np.sqrt(np.sum(x**2))
            return x, x @ (A @ x), niter
    x /= np.sqrt(np.sum(x**2))
    return x, x @ (A @ x), options.max_iters


def power_iteration2(A, x, options):
    """Power iteration method

    Performs the power iteration algorithm to find the largest eigenvalue and corresponding eigenvector of the input matrix A.

    Args:
        A (numpy.ndarray): The input square matrix.
        x (numpy.ndarray): The initial vector, assumed to be non-zero.
        options (Options): An Options object containing the maximum number of iterations and the tolerance for convergence.

    Returns:
        numpy.ndarray: The eigenvector corresponding to the largest eigenvalue.
        float: The largest eigenvalue.
        int: The number of iterations performed.
    """
    x /= np.sqrt(np.sum(x**2))
    new = A @ x
    ld = x @ new
    for niter in range(options.max_iters):
        ld1 = ld
        x[:] = new[:]
        x /= np.sqrt(np.sum(x**2))
        new = A @ x
        ld = x @ new
        if abs(ld1 - ld) <= options.tolerance:
            return x, ld, niter
    return x, ld, options.max_iters


def power_iteration3(A, x, options):
    """Power iteration method

    Performs the power iteration algorithm to find the largest eigenvalue and corresponding eigenvector of the input matrix A.

    Args:
        A (numpy.ndarray): The input square matrix.
        x (numpy.ndarray): The initial vector, assumed to be non-zero.
        options (Options): An Options object containing the maximum number of iterations and the tolerance for convergence.

    Returns:
        numpy.ndarray: The eigenvector corresponding to the largest eigenvalue.
        float: The largest eigenvalue.
        int: The number of iterations performed.
    """
    new = A @ x
    dot = x @ x
    ld = (x @ new) / dot
    for niter in range(options.max_iters):
        ld1 = ld
        x[:] = new[:]
        dot = x @ x
        if dot >= 1e150:
            x /= np.sqrt(np.sum(x**2))
            new = A @ x
            ld = x @ new
            if abs(ld1 - ld) <= options.tolerance:
                return x, ld, niter
        else:
            new = A @ x
            ld = (x @ new) / dot
            if abs(ld1 - ld) <= options.tolerance:
                x /= np.sqrt(np.sum(x**2))
                return x, ld, niter
    x /= np.sqrt(np.sum(x**2))
    return x, ld, options.max_iters


# Test data
A = np.array([[3.7, -3.6, 0.7], [-3.6, 4.3, -2.8], [0.7, -2.8, 5.4]])
options = Options(max_iters=2000, tolerance=1e-7)

x = np.array([0.3, 0.5, 0.4])
x1, ld, niter = power_iteration(A, x, options)
print(x1)
print(ld)

x = np.array([0.3, 0.5, 0.4])
x4, ld, niter = power_iteration4(A, x, options)
print(x4)
print(ld)

options.tolerance = 1e-14

x = np.array([0.3, 0.5, 0.4])
x2, ld, niter = power_iteration2(A, x, options)
print(x2)
print(ld)

x = np.array([0.3, 0.5, 0.4])
x3, ld, niter = power_iteration3(A, x, options)
print(x3)
print(ld)
