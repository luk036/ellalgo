from typing import Tuple

import numpy as np
from math import floor

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

# rand('twister',sum(100*clock))
# randn('state',sum(100*clock))


# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
class LowpassOracle:
    more_alt: bool = True

    def __init__(self, N: int, wpass: float, wstop: float, Lpsq: float, Upsq: float):
        # *********************************************************************
        # optimization parameters
        # *********************************************************************
        # rule-of-thumb discretization (from Cheney's Approximation Theory)
        m = 15 * N
        w = np.linspace(0, np.pi, m)  # omega

        # A is the matrix used to compute the power spectrum
        # A(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(N*w)]
        An = 2 * np.cos(np.outer(w, np.arange(1, N)))
        self.A = np.concatenate((np.ones((m, 1)), An), axis=1)
        self.nwpass: int = floor(wpass * (m - 1)) + 1  # end of passband
        self.nwstop: int = floor(wstop * (m - 1)) + 1  # end of stopband
        self.Lpsq = Lpsq
        self.Upsq = Upsq

    def assess_optim(self, x: Arr, Spsq: float):
        """[summary]

        Arguments:
            x (Arr): coefficients of autocorrelation
            Spsq (float): the best-so-far Sp^2

        Returns:
            [type]: [description]
        """
        # 1. nonnegative-real constraint
        n = len(x)
        self.more_alt = True

        # case 2,
        # 2. passband constraints
        N, n = self.A.shape
        for k in range(0, self.nwpass):
            v = self.A[k, :].dot(x)
            if v > self.Upsq:
                g = self.A[k, :]
                f = (v - self.Upsq, v - self.Lpsq)
                return (g, f), None

            if v < self.Lpsq:
                g = -self.A[k, :]
                f = (-v + self.Lpsq, -v + self.Upsq)
                return (g, f), None

        # case 3,
        # 3. stopband constraint
        fmax = float("-inf")
        imax = 0
        for k in range(self.nwstop, N):
            v = self.A[k, :].dot(x)
            if v > Spsq:
                g = self.A[k, :]
                f = (v - Spsq, v)
                return (g, f), None

            if v < 0:
                g = -self.A[k, :]
                f = (-v, -v + Spsq)
                return (g, f), None

            if v > fmax:
                fmax = v
                imax = k

        # case 4,
        # 1. nonnegative-real constraint on other frequences
        for k in range(self.nwpass, self.nwstop):
            v = self.A[k, :].dot(x)
            if v < 0:
                f = -v
                g = -self.A[k, :]
                return (g, f), None  # single cut

        self.more_alt = False

        # case 1 (unlikely)
        if x[0] < 0:
            g = np.zeros(n)
            g[0] = -1.0
            f = -x[0]
            return (g, f), None

        # Begin objective function
        Spsq = fmax
        f = (0.0, fmax)
        # f = 0.
        g = self.A[imax, :]
        return (g, f), Spsq


# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
def create_lowpass_case(N=48):
    """[summary]

    Keyword Arguments:
        N (int): [description] (default: {48})

    Returns:
        [type]: [description]
    """
    # wpass = 0.12 * np.pi  # end of passband
    # wstop = 0.20 * np.pi  # start of stopband

    delta0_wpass = 0.025
    delta0_wstop = 0.125
    # maximum passband ripple in dB (+/- around 0 dB)
    delta1 = 20 * np.log10(1 + delta0_wpass)
    # stopband attenuation desired in dB
    delta2 = 20 * np.log10(delta0_wstop)

    # passband 0 <= w <= w_pass
    Lp = pow(10, -delta1 / 20)
    Up = pow(10, +delta1 / 20)
    Sp = pow(10, +delta2 / 20)

    # ind_p = np.where(w <= wpass)[0]  # passband
    # Ap = A[ind_p, :]

    # # stopband (w_stop <= w)
    # ind_s = np.where(wstop <= w)[0]  # stopband
    # As = A[ind_s, :]

    # # remove redundant contraints
    # ind_beg = ind_p[-1]
    # ind_end = ind_s[0]
    # Anr = A[range(ind_beg + 1, ind_end), :]

    Lpsq = Lp * Lp
    Upsq = Up * Up
    Spsq = Sp * Sp

    omega = LowpassOracle(N, 0.12, 0.20, Lpsq, Upsq)
    return omega, Spsq
