import numpy as np
import math


class Options:
    max_iters = 2000
    tolerance = 1e-9


def power_iteration(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""
    x = x / math.sqrt(x @ x)
    for niter in range(options.max_iters):
        x1 = A @ x
        x1 = x1 / math.sqrt(x1 @ x1)
        if (
            sum(np.abs(x - x1)) <= options.tolerance
            or sum(np.abs(x + x1)) <= options.tolerance
        ):
            ld = x1 @ A @ x1
            return x1, ld, niter
        x = x1

    ld = x @ A @ x
    return x, ld, options.max_iters


def power_iteration4(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""
    x = x / sum(np.abs(x))
    for niter in range(options.max_iters):
        x1 = A @ x
        x1 = x1 / sum(np.abs(x1))
        if (
            sum(np.abs(x - x1)) <= options.tolerance
            or sum(np.abs(x + x1)) <= options.tolerance
        ):
            x1 = x1 / math.sqrt(x1 @ x1)
            ld = x1 @ A @ x1
            return x1, ld, niter
        x = x1

    x = x / math.sqrt(x @ x)
    ld = x @ A @ x
    return x, ld, options.max_iters


def power_iteration2(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""
    x /= math.sqrt(x @ x)
    new = A @ x
    ld = x @ new
    for niter in range(options.max_iters):
        ld1 = ld
        x = new
        x /= math.sqrt(x @ x)
        new = A @ x
        ld = x @ new
        if abs(ld1 - ld) <= options.tolerance:
            return x, ld, niter
    return x, ld, options.max_iters


def power_iteration3(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""

    new = A @ x
    dot = x @ x
    ld = (x @ new) / dot
    for niter in range(options.max_iters):
        ld1 = ld
        x = new
        dot = x @ x
        if dot >= 1e150:
            x /= math.sqrt(x @ x)
            new = A @ x
            ld = x @ new
            if abs(ld1 - ld) <= options.tolerance:
                return x, ld, niter
        else:
            new = A @ x
            ld = (x @ new) / dot
            if abs(ld1 - ld) <= options.tolerance:
                x /= math.sqrt(x @ x)
                return x, ld, niter
    x /= math.sqrt(dot)
    return x, ld, options.max_iters


if __name__ == "__main__":
    A = np.array([[3.7, -3.6, 0.7], [-3.6, 4.3, -2.8], [0.7, -2.8, 5.4]])
    options = Options()
    options.tolerance = 1e-7
    x, ld, niter = power_iteration(A, np.array([0.3, 0.5, 0.4]), options)

    print("1-----------------------------")
    print(x)
    print(ld)
    print(niter)

    print("4-----------------------------")

    x, ld, niter = power_iteration4(A, np.array([0.3, 0.5, 0.4]), options)
    print(x)
    print(ld)
    print(niter)

    options.tolerance = 1e-14
    print("2-----------------------------")

    x, ld, niter = power_iteration2(A, np.array([0.3, 0.5, 0.4]), options)
    print(x)
    print(ld)
    print(niter)

    print("3-----------------------------")

    x, ld, niter = power_iteration3(A, np.array([0.3, 0.5, 0.4]), options)
    print(x)
    print(ld)
    print(niter)
