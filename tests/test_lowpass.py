import time

import numpy as np
from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.oracles.lowpass_oracle import create_lowpass_case


def run_lowpass(use_parallel_cut: bool, duration=0.000001):
    """[summary]

    Arguments:
        use_parallel_cut (float): [description]

    Keyword Arguments:
        duration (float): [description] (default: {0.000001})

    Returns:
        [type]: [description]
    """
    N = 32
    r0 = np.zeros(N)  # initial xinit
    r0[0] = 0
    ellip = Ell(4.0, r0)
    ellip._helper.use_parallel_cut = use_parallel_cut
    omega, Spsq = create_lowpass_case(N)
    options = Options()
    options.max_iters = 20000
    options.tol = 1e-8
    h, _, num_iters = cutting_plane_optim(omega, ellip, Spsq, options)
    time.sleep(duration)
    # h = spectral_fact(r)
    return num_iters, h is not None


# def test_no_parallel_cut(benchmark):
#     result, feasible = benchmark(run_lowpass, False)
#     assert feasible
#     assert result >= 13334

# def test_w_parallel_cut(benchmark):
#     result, feasible = benchmark(run_lowpass, True)
#     assert feasible
#     assert result <= 568


def test_lowpass():
    """[summary]"""
    result, feasible = run_lowpass(True)
    assert feasible
    assert result >= 1083
    assert result <= 1194

