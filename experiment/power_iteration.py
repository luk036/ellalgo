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
        eps = np.linalg.norm(x - x1, np.inf)
        if eps <= options.tolerance:
            break
        eps = np.linalg.norm(x + x1, np.inf)
        if eps <= options.tolerance:
            break
        x = x1

    ld = x1 @ A @ x1
    return x1, ld, niter, eps


def calc_core2(A, x):
    x /= math.sqrt(x @ x)
    new = A @ x
    ld = x @ new
    return new, ld


def power_iteration2(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""
    new, ld = calc_core2(A, x)
    for niter in range(options.max_iters):
        ld1 = ld
        x = new
        new, ld = calc_core2(A, x)
        eps = abs(ld1 - ld)
        if eps <= options.tolerance:
            break
    return x, ld, niter, eps


def calc_core3(A, x):
    new = A @ x
    dot = x @ x
    ld = (x @ new) / dot
    return new, dot, ld


def power_iteration3(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""

    new, dot, ld = calc_core3(A, x)
    for niter in range(options.max_iters):
        ld1 = ld
        x = new
        xmax = max(x)
        xmin = min(x)
        if xmax >= 1e100 or xmin <= -1e100:
            x /= 1e100
        new, dot, ld = calc_core3(A, x)
        eps = abs(ld1 - ld)
        if eps <= options.tolerance:
            break

    x /= math.sqrt(dot)
    return x, ld, niter, eps


def power_iteration4(A: np.ndarray, x: np.ndarray, options):
    """Power iteration method"""
    x = x / max(np.abs(x))
    for niter in range(options.max_iters):
        x1 = A @ x
        x1 = x1 / max(np.abs(x1))
        eps = np.linalg.norm(x - x1, np.inf)
        if eps <= options.tolerance:
            break
        eps = np.linalg.norm(x + x1, np.inf)
        if eps <= options.tolerance:
            break
        x = x1

    x1 = x1 / math.sqrt(x1 @ x1)
    ld = x1 @ A @ x1
    return x1, ld, niter, eps


if __name__ == "__main__":
    A = np.array([[3.7, -4.5], [4.3, -5.9]])
    x, ld, niter, eps = power_iteration(A, np.array([1.1, 0.0]), Options())

    print("1-----------------------------")
    print(x)
    print(ld)
    print(niter)
    print(eps)

    print("2-----------------------------")

    x, ld, niter, eps = power_iteration2(A, np.array([1.1, 0.0]), Options())
    print(x)
    print(ld)
    print(niter)
    print(eps)

    print("3-----------------------------")

    x, ld, niter, eps = power_iteration3(A, np.array([1.1, 0.0]), Options())
    print(x)
    print(ld)
    print(niter)
    print(eps)

    print("4-----------------------------")

    x, ld, niter, eps = power_iteration4(A, np.array([1.1, 0.0]), Options())
    print(x)
    print(ld)
    print(niter)
    print(eps)
