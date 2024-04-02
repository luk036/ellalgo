import numpy as np

from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.oracles.lowpass_oracle import create_lowpass_case


def run_lowpass(use_parallel_cut: bool):
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
    ellip = Ell(40.0, r0)
    ellip._helper.use_parallel_cut = use_parallel_cut
    omega, Spsq = create_lowpass_case(N)
    options = Options()
    options.max_iters = 50000
    options.tolerance = 1e-14
    h, _, num_iters = cutting_plane_optim(omega, ellip, Spsq, options)
    return h is not None, num_iters


def test_lowpass():
    """Test the lowpass case with parallel cut"""
    feasible, num_iters = run_lowpass(True)
    assert feasible
    assert num_iters >= 23000
    assert num_iters <= 24000
