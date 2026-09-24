"""
FIR low-pass filter design oracle via spectral factorization.

This module implements the approach from:
    S.-P. Wu, S. Boyd, and L. Vandenberghe, "FIR Filter Design via Spectral
    Factorization and Convex Optimization"

The `LowpassOracle` class formulates the FIR filter design problem as a convex
optimization over the auto-correlation coefficients. It checks passband ripple
and stopband attenuation constraints using a pre-computed spectrum matrix for
efficient frequency response evaluation at discretized frequency points.

Key methods:
    - assess_feas: Check whether filter coefficients meet passband/stopband specs.
    - assess_optim: Assess optimality, returning the maximum stopband response.

Also provides `create_lowpass_case()` for a standard test case with typical
parameters (passband 0-0.12π, stopband 0.20-π, ±0.025dB ripple).
"""

from math import floor
from typing import Callable, Optional, Tuple, Union

import numpy as np

from ellalgo.ell_typing import CutChoice, OracleOptim
from ellalgo.round_robin import RoundRobin

Arr = np.ndarray
ParallelCut = Tuple[Arr, CutChoice]


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


class _Band:
    """A frequency band scanned for its first constraint violation.

    ``upper`` is a float, a zero-argument callable returning the current bound
    (the stopband bound tracks gamma), or ``None`` for a one-sided
    non-negativity band. ``track_max`` enables the stopband peak tracking used
    by :meth:`LowpassOracle.assess_optim`.
    """

    __slots__ = ("start", "stop", "lower", "upper", "cursor", "track_max")

    def __init__(
        self,
        start: int,
        stop: int,
        lower: float,
        upper: Union[float, Callable[[], float], None],
        cursor: RoundRobin,
        track_max: bool = False,
    ) -> None:
        self.start = start
        self.stop = stop
        self.lower = lower
        self.upper = upper
        self.cursor = cursor
        self.track_max = track_max


# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
class LowpassOracle(OracleOptim):
    idx1: RoundRobin

    def __init__(
        self,
        ndim: int,
        wpass: float,
        wstop: float,
        lp_sq: float,
        up_sq: float,
        sp_sq: float,
    ):
        """
        Initializes a LowpassOracle object with the given parameters.

        Args:
            ndim (int): The number of FIR coefficients (including the zeroth).
            wpass (float): The end of the passband.
            wstop (float): The end of the stopband.
            lp_sq (float): The lower bound on the squared magnitude frequency response in the passband.
            up_sq (float): The upper bound on the squared magnitude frequency response in the passband.
            sp_sq (float): The upper bound on the squared magnitude frequency response in the stopband.

        Attributes:
            spectrum (np.ndarray): The matrix used to compute the power spectrum.
            nwpass (int): The index of the end of the passband.
            nwstop (int): The index of the end of the stopband.
            lp_sq (float): The lower bound on the squared magnitude frequency response in the passband.
            up_sq (float): The upper bound on the squared magnitude frequency response in the passband.
            sp_sq (float): The upper bound on the squared magnitude frequency response in the stopband.
            idx1 (RoundRobin): Round-robin cursor for the passband.
            idx2 (RoundRobin): Round-robin cursor for the transition band.
            idx3 (RoundRobin): Round-robin cursor for the stopband.
            fmax (float): The maximum value of the squared magnitude frequency response.
            kmax (int): The index of the maximum value of the squared magnitude frequency response.
        """
        # *********************************************************************
        # optimization parameters
        # *********************************************************************
        # rule-of-thumb discretization (from Cheney's Approximation Theory)
        mdim = 15 * ndim  # Number of frequency points to evaluate
        w = np.linspace(0, np.pi, mdim)  # omega (frequency points from 0 to π)

        # spectrum is the matrix used to compute the power spectrum
        # spectrum(w,:) = [1 2*cos(w) 2*cos(2*w) ... 2*cos(mdim*w)]
        # This creates a matrix where each row corresponds to a frequency point,
        # and each column contains the cosine terms for that frequency
        temp = 2 * np.cos(np.outer(w, np.arange(1, ndim)))
        self.spectrum = np.concatenate((np.ones((mdim, 1)), temp), axis=1)

        # Convert normalized frequency bounds to array indices
        self.nwpass: int = floor(wpass * (mdim - 1)) + 1  # end of passband
        self.nwstop: int = floor(wstop * (mdim - 1)) + 1  # end of stopband

        # Store the squared magnitude bounds
        self.lp_sq = lp_sq  # Lower bound for passband (squared)
        self.up_sq = up_sq  # Upper bound for passband (squared)
        self.sp_sq = sp_sq  # Upper bound for stopband (squared)

        # Initialize indices for round-robin checking of frequency points
        self.idx1 = RoundRobin(self.nwpass, start=0)  # passband: [0, nwpass)
        self.idx2 = RoundRobin(
            self.nwstop, lo=self.nwpass, start=self.nwpass
        )  # transition band
        self.idx3 = RoundRobin(mdim, lo=self.nwstop, start=self.nwstop)  # stopband

        # Variables to track maximum response in stopband
        self.fmax = float("-inf")  # Maximum response value found
        self.kmax = 0  # Index where maximum response occurs

        self._bands = (
            _Band(0, self.nwpass, self.lp_sq, self.up_sq, self.idx1),
            _Band(
                self.nwstop,
                self.spectrum.shape[0],
                0.0,
                lambda: self.sp_sq,
                self.idx3,
                track_max=True,
            ),
            _Band(self.nwpass, self.nwstop, 0.0, None, self.idx2),
        )

    def assess_feas(self, x: Arr) -> Optional[ParallelCut]:
        """
        Assess whether the given filter coefficients meet the design specifications.

        Scans the passband, stopband, and transition band (in that order) and
        returns the first violating cut; each band resumes its round-robin
        cursor so load is distributed across calls.

        Args:
            x (Arr): The filter coefficients (autocorrelation coefficients)

        Returns:
            Optional[ParallelCut]:
                - None if all specifications are met
                - A tuple containing the gradient of the violating constraint
                  and the violation amount (a pair for two-sided bands, a scalar
                  for the non-negativity band)
        """
        for band in self._bands:
            cut = self._band_cut(band, x)
            if cut is not None:
                return cut

        if x[0] < 0:
            grad = np.zeros(self.spectrum.shape[1])
            grad[0] = -1.0
            return grad, -x[0]

        return None

    def _band_cut(self, band: _Band, x: Arr) -> Optional[ParallelCut]:
        """Return the first violation cut in ``band``, or ``None``.

        Two-sided bands return ``(g, (lo_viol, up_viol))``; a one-sided band
        (``upper is None``) returns ``(g, violation)``. Stopband peak tracking
        (``fmax``/``kmax``) is refreshed here when no violation is found.
        """
        upper = band.upper() if callable(band.upper) else band.upper
        if band.track_max:
            self.fmax = float("-inf")
            self.kmax = 0

        v = self.spectrum[band.start : band.stop] @ x
        if v.size == 0:
            return None

        offset = band.cursor.peek_next() - band.start
        if upper is None:
            j = self._first_rotated(v < band.lower, offset)
            if j < 0:
                return None
            idx = band.start + j
            band.cursor.seek(idx)
            return -self.spectrum[idx, :], band.lower - v[j]

        over = v > upper
        j = self._first_rotated(over | (v < band.lower), offset)
        if j < 0:
            if band.track_max:
                vmax = v.max()
                self.fmax = float(vmax)
                self.kmax = band.start + self._first_rotated(v == vmax, offset)
            return None
        idx = band.start + j
        band.cursor.seek(idx)
        col = self.spectrum[idx, :]
        val = v[j]
        if over[j]:
            return col, (val - upper, val - band.lower)
        return -col, (band.lower - val, upper - val)

    @staticmethod
    def _first_rotated(mask: np.ndarray, offset: int) -> int:
        """Return the first ``True`` index of ``mask`` starting at ``offset``.

        The round-robin scan visits ``offset, offset + 1, ..., n - 1, 0, ...,
        offset - 1``; this locates the first ``True`` in that order without
        materialising the rotated index array.

        Args:
            mask: Boolean array of per-point violation flags.
            offset: Starting position of the round-robin scan.

        Returns:
            Index of the first ``True`` in round-robin order, or -1 if none.

        Examples:
            >>> import numpy as np
            >>> LowpassOracle._first_rotated(np.array([False, True, False]), 2)
            1
            >>> LowpassOracle._first_rotated(np.array([False, False]), 0)
            -1
        """
        tail = mask[offset:]
        if tail.any():
            return offset + int(tail.argmax())
        head = mask[:offset]
        return int(head.argmax()) if head.any() else -1

    def assess_optim(
        self, xc: Arr, gamma: float
    ) -> Tuple[ParallelCut, Optional[float]]:
        """
        Assess the optimality of the current filter coefficients for the stopband.

        First checks feasibility using assess_feas. If feasible, returns information
        about the maximum response in the stopband which can be used to further
        optimize the filter design.

        Args:
            xc (Arr): The filter coefficients (autocorrelation coefficients)
            gamma (float): The current best stopband attenuation value to beat

        Returns:
            tuple: A tuple containing:
                - A tuple of (gradient, (lower, upper)) for the maximum stopband response
                - The maximum stopband response value (or None if not feasible)
        """
        # Update the stopband bound
        self.sp_sq = gamma

        # First check feasibility
        if cut := self.assess_feas(xc):
            return cut, None  # Return feasibility cut and no objective value

        # If feasible, return information about the maximum stopband response
        return (self.spectrum[self.kmax, :], (0.0, self.fmax)), self.fmax


# *********************************************************************
# filter specs (for a low-pass filter)
# *********************************************************************
# number of FIR coefficients (including zeroth)
def create_lowpass_case(ndim: int = 48) -> "LowpassOracle":
    """
    Creates a standard low-pass filter design case with typical parameters.

    Sets up a LowpassOracle instance with commonly used specifications:
    - Passband edge at 0.12π
    - Stopband edge at 0.20π
    - Passband ripple of ±0.025 dB
    - Stopband attenuation of 0.125

    Args:
        ndim (int, optional): Number of filter coefficients. Defaults to 48.

    Returns:
        LowpassOracle: An initialized LowpassOracle instance with standard parameters
    """
    # Define normalized frequency tolerances
    delta0_wpass = 0.025  # Passband ripple tolerance
    delta0_wstop = 0.125  # Stopband attenuation tolerance

    # Convert to dB scale for calculations
    delta1 = 20 * np.log10(1 + delta0_wpass)  # Passband ripple in dB
    delta2 = 20 * np.log10(delta0_wstop)  # Stopband attenuation in dB

    # Convert dB specifications to linear scale
    low_pass = pow(10, -delta1 / 20)  # Lower passband bound
    up_pass = pow(10, +delta1 / 20)  # Upper passband bound
    stop_pass = pow(10, +delta2 / 20)  # Stopband bound

    # Square the bounds for use with squared magnitude response
    lp_sq = low_pass * low_pass
    up_sq = up_pass * up_pass
    sp_sq = stop_pass * stop_pass

    # Create and return LowpassOracle instance with these parameters
    return LowpassOracle(ndim, 0.12, 0.20, lp_sq, up_sq, sp_sq)
