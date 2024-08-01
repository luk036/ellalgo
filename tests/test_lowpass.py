"""
Test Lowpass
"""

import numpy as np

from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.oracles.lowpass_oracle import create_lowpass_case


def run_lowpass(use_parallel_cut: bool):
    """
    This Python function runs a lowpass filter optimization using cutting plane method.

    :param use_parallel_cut: The `use_parallel_cut` parameter is a boolean flag that determines whether
        to use parallel cut or not in the `run_lowpass` function
    :type use_parallel_cut: bool
    :return: The function `run_lowpass` returns a tuple containing a boolean value indicating whether
        `h` is not None, and the number of iterations `num_iters`.
    """
    ndim = 32
    r0 = np.zeros(ndim)  # initial xinit
    r0[0] = 0
    ellip = Ell(40.0, r0)
    ellip.helper.use_parallel_cut = use_parallel_cut
    omega = create_lowpass_case(ndim)
    Spsq = omega.sp_sq
    options = Options()
    options.max_iters = 50000
    options.tolerance = 1e-14
    h, _, num_iters = cutting_plane_optim(omega, ellip, Spsq, options)
    return h is not None, num_iters


def test_lowpass():
    """
    The `test_lowpass` function tests the lowpass case with parallel cut by checking if the solution is
    feasible and the number of iterations falls within a specific range.
    """
    feasible, num_iters = run_lowpass(True)
    assert feasible
    assert num_iters >= 23000
    assert num_iters <= 24000
