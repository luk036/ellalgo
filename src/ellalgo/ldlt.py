import numpy as np

Arr = np.ndarray
Mat = np.ndarray


def ldlt_rank1update(tri_lower: Mat, inv_diag: Arr, alpha: int, vec: Arr):
    told = 1 / alpha  # initially
    ndim = len(vec)
    m = ndim - 1
    for j in range(m):
        p = vec[j]
        temp = p * inv_diag[j]
        tnew = told + p * temp
        beta = temp / tnew
        inv_diag[j] *= told / tnew  # update invD
        for k in range(j + 1, ndim):
            vec[k] -= p * tri_lower[k, j]
            tri_lower[k, j] += beta * vec[k]
        told = tnew

    p = vec[m]
    temp = p * inv_diag[m]
    tnew = told + p * temp
    inv_diag[m] *= told / tnew  # update invD


def ldlt_solve1(tri_lower: Mat, vec: Arr):
    "perform L^-1 * v"
    ndim = len(vec)
    for i in range(ndim):
        for j in range(i + 1, ndim):
            vec[j] -= vec[i] * tri_lower[j, i]


def ldlt_solve2(tri_lower: Mat, vec: Arr):
    "perform L^-1 * v"
    ndim = len(vec)
    for i in range(1, ndim):
        for j in range(i):
            tri_lower[j, i] = tri_lower[i, j] * vec[j]
            # keep for rank-one update
            vec[i] -= tri_lower[j, i]
