import numpy as np
import pytest
from pytest import approx

from ellalgo.oracles.spectral_fact import inverse_spectral_fact, spectral_fact


def test_spectral_fact_empty_input() -> None:
    """Test spectral_fact raises ValueError for empty input (line 53)."""
    with pytest.raises(ValueError, match="Input array cannot be empty"):
        spectral_fact(np.array([]))


def test_spectral_fact_nan_input() -> None:
    """Test spectral_fact raises ValueError for NaN input (line 56)."""
    with pytest.raises(ValueError, match="non-finite"):
        spectral_fact(np.array([1.0, np.nan, 0.5]))


def test_spectral_fact_inf_input() -> None:
    """Test spectral_fact raises ValueError for infinite input (line 56)."""
    with pytest.raises(ValueError, match="non-finite"):
        spectral_fact(np.array([1.0, np.inf, 0.5]))


def test_spectral_fact_runtime_error() -> None:
    """Test spectral_fact raises RuntimeError for invalid auto-correlation (line 86)."""
    # Input producing negative frequency response below the -1e-4 threshold
    r = np.array([-1.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="Spectral factorization failed"):
        spectral_fact(r.reshape(-1, 1))


def test_spectral_fact() -> None:
    h = np.array(
        [
            0.76006445,
            0.54101887,
            0.42012073,
            0.3157191,
            0.10665804,
            0.04326203,
            0.01315678,
        ]
    )
    r = inverse_spectral_fact(h)
    h2 = spectral_fact(r)
    assert len(h) == len(h2)
    print(h2)
    assert h2 == approx(h)
