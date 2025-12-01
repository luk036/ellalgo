import time
from typing import Any, Tuple

import numpy as np

from ellalgo.cutting_plane import Options, cutting_plane_optim
from ellalgo.ell import Ell
from ellalgo.oracles.lowpass_oracle import create_lowpass_case


def run_lowpass(use_parallel_cut: bool, duration: float = 0.000001) -> Tuple[int, bool]:
    """[summary]

    Arguments:
        use_parallel_cut (float): [description]

    Keyword Arguments:
        duration (float): [description] (default: {0.000001})

    Returns:
        [type]: [description]
    """
    ndim = 32
    r0 = np.zeros(ndim)  # initial xinit
    r0[0] = 0
    ellip = Ell(4.0, r0)
    ellip.helper.use_parallel_cut = use_parallel_cut
    omega = create_lowpass_case(ndim)
    Spsq = omega.sp_sq
    options = Options()
    options.max_iters = 50000
    options.tolerance = 1e-14
    h, _, num_iters = cutting_plane_optim(omega, ellip, Spsq, options)
    time.sleep(duration)
    return num_iters, h is not None


def test_lowpass(benchmark: Any) -> None:
    """Test the lowpass case with parallel cut"""
    result, feasible = benchmark(run_lowpass, True)
    assert feasible
    assert result >= 12300
    assert result <= 12600


def test_no_parallel_cut(benchmark: Any) -> None:
    """Test the lowpass case with no parallel cut"""
    result, feasible = benchmark(run_lowpass, False)
    assert feasible
    assert result >= 26399
    assert result <= 30000
