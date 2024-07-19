from math import floor
from typing import Optional, Tuple, Union
from ellalgo.ell_typing import OracleOptim

import numpy as np

Arr = np.ndarray
ParallelCut = Tuple[Arr, Union[float, Tuple[float, float]]]


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
class LowpassOracle(OracleOptim):
    # more_alt: bool = True
    idx1: int = 0

    def __init__(
        self,
        ndim: int,
        wpass: float,
        wstop: float,
        lp_sq: float,
        up_sq: float,
        sp_sq: float,
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
        self.sp_sq = sp_sq
        self.idx1 = 0
        self.idx2 = self.nwpass
        self.idx3 = self.nwstop
        self.fmax = float("-inf")
        self.kmax = 0

    def assess_feas(self, x: Arr) -> Optional[ParallelCut]:
        """[summary]

        Arguments:
            x (Arr): coefficients of autocorrelation
            sp_sq (float): the best-so-far stop_pass^2

        Returns:
            [type]: [description]
        """
        # self.more_alt = True

        mdim, ndim = self.spectrum.shape
        for _ in range(self.nwpass):
            self.idx1 += 1
            if self.idx1 == self.nwpass:
                self.idx1 = 0  # round robin
            col_k = self.spectrum[self.idx1, :]
            v = col_k.dot(x)
            if v > self.up_sq:
                f = (v - self.up_sq, v - self.lp_sq)
                return col_k, f
            if v < self.lp_sq:
                f = (-v + self.lp_sq, -v + self.up_sq)
                return -col_k, f

        self.fmax = float("-inf")
        self.kmax = 0
        for _ in range(self.nwstop, mdim):
            self.idx3 += 1
            if self.idx3 == mdim:
                self.idx3 = self.nwstop  # round robin
            col_k = self.spectrum[self.idx3, :]
            v = col_k.dot(x)
            if v > self.sp_sq:
                return col_k, (v - self.sp_sq, v)
            if v < 0:
                return -col_k, (-v, -v + self.sp_sq)
            if v > self.fmax:
                self.fmax = v
                self.kmax = self.idx3

        # case 4,
        # 1. nonnegative-real constraint on other frequences
        for _ in range(self.nwpass, self.nwstop):
            self.idx2 += 1
            if self.idx2 == self.nwstop:
                self.idx2 = self.nwpass  # round robin
            col_k = self.spectrum[self.idx2, :]
            v = col_k.dot(x)
            if v < 0:
                return -col_k, -v  # single cut

        # self.more_alt = False

        # case 1 (unlikely)
        if x[0] < 0:
            grad = np.zeros(ndim)
            grad[0] = -1.0
            return grad, -x[0]

        return None

    def assess_optim(self, x: Arr, sp_sq: float):
        """[summary]

        Arguments:
            x (Arr): coefficients of autocorrelation
            sp_sq (float): the best-so-far stop_pass^2

        Returns:
            [type]: [description]
        """
        self.sp_sq = sp_sq
        if cut := self.assess_feas(x):
            return cut, None
        # Begin objective function
        return (self.spectrum[self.kmax, :], (0.0, self.fmax)), self.fmax


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

    return LowpassOracle(ndim, 0.12, 0.20, lp_sq, up_sq, sp_sq)
