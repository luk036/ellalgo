from typing import Union

import numpy as np

Arr = Union[np.ndarray]
Mat = Union[np.ndarray]


def ldlt_rank1update(low: Mat, inv_diag: Arr, alpha: int, vec: Arr):
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
            vec[k] -= p * low[k, j]
            low[k, j] += beta * vec[k]
        told = tnew

    p = vec[m]
    temp = p * inv_diag[m]
    tnew = told + p * temp
    inv_diag[m] *= told / tnew  # update invD


def ldlt_solve1(low: Mat, vec: Arr):
    "perform L^-1 * v"
    ndim = len(vec)
    for i in range(ndim):
        for j in range(i + 1, ndim):
            vec[j] -= vec[i] * low[i, j]


def ldlt_solve2(low: Mat, vec: Arr):
    "perform L^-1 * v"
    ndim = len(vec)
    for i in range(1, ndim):
        for j in range(i):
            low[j, i] = low[i, j] * vec[j]
            # keep for rank-one update
            vec[i] -= low[j, i]
