from math import floor
from typing import Tuple

import numpy as np

Arr = np.ndarray
Cut = Tuple[Arr, float]


# Modified from CVX code by Almir Mutapcic in 2006.
# Adapted in 2010 for impulse response peak-minimization by convex iteration
# by Christine Law.
#
# "FIR Filter Design via Spectral Factorization and Convex Optimization"
# by S.-P. Wu, S. Boyd, and L. Vandenberghe
#
# Designs an FIR lowpass filter using spectral factorization method with
# constraint on maximum passband ripple and stopband attenuation:
#
#   minimize   max |H(w)|                      for w in stopband
#       s.t.   1/delta <= |H(w)| <= delta      for w in passband
#
# We change variables via spectral factorization method and get:
#
#   minimize   max R(w)                          for w in stopband
#       s.t.   (1/delta)**2 <= R(w) <= delta**2  for w in passband
#              R(w) >= 0                         for all w
#
# where R(w) is squared magnitude frequency response
# (and Fourier transform of autocorrelation coefficients r).
# Variables are coeffients r and gra = hh' where h is impulse response.
# delta is allowed passband ripple.
# This is a convex problem (can be formulated as an SDP after sampling).


# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
class LowpassOracle:
    more_alt: bool = True

    def __init__(
        self, ndim: int, wpass: float, wstop: float, lp_sq: float, up_sq: float
    ):
        # *********************************************************************
        # optimization parameters
        # *********************************************************************
        # rule-of-thumb discretization (from Cheney's Approximation Theory)
        mdim = 15 * ndim
        w = np.linspace(0, np.pi, mdim)  # omega

        # spectrum is the matrix used to compute the power spectrum
        # spectrum(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(mdim*w)]
        temp = 2 * np.cos(np.outer(w, np.arange(1, ndim)))
        self.spectrum = np.concatenate((np.ones((mdim, 1)), temp), axis=1)
        self.nwpass: int = floor(wpass * (mdim - 1)) + 1  # end of passband
        self.nwstop: int = floor(wstop * (mdim - 1)) + 1  # end of stopband
        self.lp_sq = lp_sq
        self.up_sq = up_sq

    def assess_optim(self, x: Arr, sp_sq: float):
        """[summary]

        Arguments:
            x (Arr): coefficients of autocorrelation
            sp_sq (float): the best-so-far stop_pass^2

        Returns:
            [type]: [description]
        """
        # 1. nonnegative-real constraint
        self.more_alt = True

        # case 2,
        # 2. passband constraints
        mdim, ndim = self.spectrum.shape
        for k in range(self.nwpass):
            col_k = self.spectrum[k, :]
            v = col_k.dot(x)
            if v > self.up_sq:
                f = (v - self.up_sq, v - self.lp_sq)
                return (col_k, f), None
            if v < self.lp_sq:
                f = (-v + self.lp_sq, -v + self.up_sq)
                return (-col_k, f), None

        # case 3,
        # 3. stopband constraint
        fmax = float("-inf")
        kmax = 0
        for k in range(self.nwstop, mdim):
            col_k = self.spectrum[k, :]
            v = col_k.dot(x)
            if v > sp_sq:
                return (col_k, (v - sp_sq, v)), None
            if v < 0:
                return (-col_k, (-v, -v + sp_sq)), None
            if v > fmax:
                fmax = v
                kmax = k

        # case 4,
        # 1. nonnegative-real constraint on other frequences
        for k in range(self.nwpass, self.nwstop):
            col_k = self.spectrum[k, :]
            v = col_k.dot(x)
            if v < 0:
                return (-col_k, -v), None  # single cut

        self.more_alt = False

        # case 1 (unlikely)
        if x[0] < 0:
            grad = np.zeros(ndim)
            grad[0] = -1.0
            return (grad, -x[0]), None

        # Begin objective function
        return (self.spectrum[kmax, :], (0.0, fmax)), fmax


# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
def create_lowpass_case(ndim=48):
    """[summary]

    Keyword Arguments:
        mdim (int): [description] (default: {48})

    Returns:
        [type]: [description]
    """
    delta0_wpass = 0.025
    delta0_wstop = 0.125
    # maximum passband ripple in dB (+/- around 0 dB)
    delta1 = 20 * np.log10(1 + delta0_wpass)
    # stopband attenuation desired in dB
    delta2 = 20 * np.log10(delta0_wstop)

    # passband 0 <= w <= w_pass
    low_pass = pow(10, -delta1 / 20)
    up_pass = pow(10, +delta1 / 20)
    stop_pass = pow(10, +delta2 / 20)

    lp_sq = low_pass * low_pass
    up_sq = up_pass * up_pass
    sp_sq = stop_pass * stop_pass

    omega = LowpassOracle(ndim, 0.12, 0.20, lp_sq, up_sq)
    return omega, sp_sq
