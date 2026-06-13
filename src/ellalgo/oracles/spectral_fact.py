"""
Spectral factorization for minimum-phase impulse response computation.

Implements the Kolmogorov 1939 spectral factorization approach as described
in A. Papoulis, "Signal Analysis" (pp. 232-233). This is used by the
LowpassOracle to convert between auto-correlation coefficients and the
minimum-phase impulse response of an FIR filter.

Functions:
    - spectral_fact(r): Compute minimum-phase impulse response from auto-correlation.
    - inverse_spectral_fact(h): Reconstruct auto-correlation from impulse response.

The spectral factorization pipeline:
    auto-correlation → oversampling → log(|R(w)|) → Hilbert transform →
    complex log-spectrum → IFFT → impulse response
"""

import numpy as np

__all__ = ["spectral_fact", "inverse_spectral_fact"]


def spectral_fact(r: np.ndarray) -> np.ndarray:
    """Computes the minimum-phase impulse response satisfying a given auto-correlation.

    This function implements the Kolmogorov 1939 approach to spectral
    factorization, as described in pp. 232-233 of "Signal Analysis" by
    A. Papoulis.

    Args:
        r (numpy.ndarray): The top-half of the auto-correlation coefficients,
            starting from the 0th element to the end of the auto-correlation.
            This should be passed in as a column vector.

    Returns:
        numpy.ndarray: The impulse response that gives the desired auto-correlation.

    Raises:
        ValueError: If the input array is empty or contains invalid values.
        RuntimeError: If numerical errors occur during spectral factorization (e.g., log of negative numbers, FFT errors).

    Examples:
        >>> r = np.array([1.0, 0.5, 0.2])
        >>> h = spectral_fact(r.reshape(-1, 1))
        >>> isinstance(h, np.ndarray)
        True
        >>> h.shape == (r.shape[0], r.shape[0])
        True
    """
    try:
        # Validate input
        if len(r) == 0:
            raise ValueError("Input array cannot be empty")

        if not np.all(np.isfinite(r)):
            raise ValueError("Input array contains non-finite values (NaN or infinity)")

        # length of the impulse response sequence
        n = len(r)

        # over-sampling factor
        mult_factor = 100  # should have mult_factor*(n) >> n
        m = mult_factor * n

        # computation method:
        # H(exp(jTw)) = alpha(w) + j*phi(w)
        # where alpha(w) = 1/2*ln(R(w)) and phi(w) = Hilbert_trans(alpha(w))

        # compute 1/2*ln(R(w))
        # w = 2*pi*[0:m-1]/m
        w = np.linspace(0, 2 * np.pi, m, endpoint=False)
        # R = [ones(m, 1) 2*cos(kron(w', [1:n-1]))]*r
        Bn = np.outer(w, np.arange(1, n))
        An = 2 * np.cos(Bn)
        R = np.hstack((np.ones((m, 1)), An)) @ r  # NOQA

        # Check for negative or zero values before taking log
        # Allow small negative values due to numerical precision issues
        min_val = np.min(R)
        if min_val <= 0:
            # If the minimum is very close to zero (numerical precision issue),
            # clamp to a small positive value
            if min_val > -1e-4:
                R = np.maximum(R, 1e-10)
            else:
                raise RuntimeError(
                    f"Spectral factorization failed: frequency response contains non-positive values. "
                    f"This indicates the input auto-correlation may not be valid. "
                    f"Minimum value: {min_val:.6e}, Negative values: {np.sum(R < 0)}"
                )

        # alpha = ne.evaluate("0.5 * log(abs(R))")
        alpha = 0.5 * np.log(np.abs(R))

        # find the Hilbert transform
        alphatmp = np.fft.fft(alpha)
        # alphatmp(floor(m/2)+1: m) = -alphatmp(floor(m/2)+1: m)
        ind = int(m / 2)  # python3 need int()
        alphatmp[ind:m] = -alphatmp[ind:m]
        alphatmp[0] = 0
        alphatmp[ind] = 0
        phi = np.real(np.fft.ifft(1j * alphatmp))

        # now retrieve the original sampling
        # index = find(np.reminder([0:m-1], mult_factor) == 0)
        index = np.arange(0, m, step=int(mult_factor))
        alpha1 = alpha[index]
        phi1 = phi[index]

        # compute the impulse response (inverse Fourier transform)
        h = np.real(np.fft.ifft(np.exp(alpha1 + 1j * phi1), n))

        return h

    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input for spectral factorization: {e}")
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Linear algebra error during spectral factorization: {e}")
    except Exception as e:
        raise RuntimeError(f"Spectral factorization failed with unexpected error: {e}")


def inverse_spectral_fact(h: np.ndarray) -> np.ndarray:
    """
    Computes the auto-correlation sequence from the given impulse response.

    Arguments:
        h (numpy.ndarray): The impulse response sequence.

    Returns:
        numpy.ndarray: The auto-correlation sequence, where the length is the same as the input impulse response.

    Examples:
        >>> h = np.array([1.0, 0.5, 0.2])
        >>> r = inverse_spectral_fact(h)
        >>> isinstance(r, np.ndarray)
        True
        >>> r.shape == (len(h),)
        True
    """
    n = len(h)
    # Take bottom-half of the auto-corelation function due to symmetry ???
    return np.convolve(h, h[::-1])[n - 1 :]
    # r = np.zeros(n)
    # for t in range(n):
    #     r[t] = h[t:] @ h[: n - t]
    # return r


# if __name__ == "__main__":
#     r = np.random.rand(20)
#     h = spectral_fact(r)
#     print(h)
