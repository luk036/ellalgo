"""
Property-based tests for the Ell class using Hypothesis.

These tests verify that the ellipsoid update methods satisfy mathematical properties
that should hold regardless of the specific input values.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pytest import approx

from ellalgo.ell import Ell
from ellalgo.ell_config import CutStatus


# Helper strategies for generating test data
@st.composite
def valid_ellipsoid_strategy(draw):
    """Generate valid ellipsoid parameters."""
    ndim = draw(st.integers(min_value=2, max_value=10))  # EllCalc requires ndim >= 2

    # Generate center point
    xc = draw(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=ndim, max_size=ndim)
    )
    xc = np.array(xc)

    # Generate kappa (positive)
    kappa = draw(st.floats(min_value=0.001, max_value=10))

    return kappa, xc


@st.composite
def valid_cut_strategy(draw, ndim):
    """Generate valid cuts for given dimension."""
    # Ensure ndim >= 2 for EllCalc
    assume(ndim >= 2)

    # Generate gradient (non-zero)
    grad = draw(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=ndim, max_size=ndim)
    )
    grad = np.array(grad)

    # Ensure gradient is not zero
    assume(np.any(grad != 0))

    # Generate beta (non-negative for bias cuts)
    beta = draw(st.floats(min_value=0.0, max_value=5))

    return grad, beta


@st.composite
def valid_parallel_cut_strategy(draw, ndim):
    """Generate valid parallel cuts for given dimension."""
    # Ensure ndim >= 2 for EllCalc
    assume(ndim >= 2)

    # Generate gradient (non-zero)
    grad = draw(
        st.lists(st.floats(min_value=-10, max_value=10), min_size=ndim, max_size=ndim)
    )
    grad = np.array(grad)

    # Ensure gradient is not zero
    assume(np.any(grad != 0))

    # Generate beta list (beta1, beta2) where beta1 < beta2
    # Both should be >= 0 for bias cuts
    beta1 = draw(st.floats(min_value=0.0, max_value=2.0))
    beta2 = draw(st.floats(min_value=beta1 + 0.01, max_value=5.0))

    return grad, [beta1, beta2]


class TestEllipsoidProperties:
    """Property-based tests for ellipsoid mathematical properties."""

    @given(valid_ellipsoid_strategy())
    def test_ellipsoid_initialization_properties(self, ell_params):
        """Test that ellipsoid initialization preserves basic properties."""
        kappa, xc = ell_params
        ell = Ell(kappa, xc)

        # Center should be preserved
        assert np.allclose(ell.xc(), xc)

        # Initial tsq should be 0
        assert ell.tsq() == 0.0

        # kappa should be preserved for scalar initialization
        if isinstance(kappa, (int, float)):
            assert ell._kappa == kappa

        # Shape matrix should be positive definite
        eigenvalues = np.linalg.eigvals(ell._mq)
        assert np.all(eigenvalues > 0)

    @given(valid_ellipsoid_strategy(), valid_cut_strategy(4))
    @settings(max_examples=100)
    def test_central_cut_preserves_positive_definiteness(self, ell_params, cut):
        """Test that central cuts preserve positive definiteness of shape matrix."""
        kappa, xc = ell_params

        # Create 4D ellipsoid for consistent testing
        xc_4d = np.zeros(4)
        ell = Ell(kappa, xc_4d)

        grad, beta = cut

        status = ell.update_central_cut((grad, 0.0))  # True central cut

        if status == CutStatus.Success:
            # Shape matrix should remain positive semi-definite
            eigenvalues = np.linalg.eigvals(ell._mq)
            assert np.all(eigenvalues >= 0)

            # kappa should remain positive
            assert ell._kappa > 0

    @given(valid_ellipsoid_strategy(), valid_cut_strategy(4))
    @settings(max_examples=100)
    def test_bias_cut_preserves_positive_definiteness(self, ell_params, cut):
        """Test that bias cuts preserve positive definiteness of shape matrix."""
        kappa, xc = ell_params

        # Create 4D ellipsoid for consistent testing
        xc_4d = np.zeros(4)
        ell = Ell(kappa, xc_4d)

        grad, beta = cut

        status = ell.update_bias_cut((grad, beta))

        # Status should be valid (not an error)
        assert status in [CutStatus.Success, CutStatus.NoSoln, CutStatus.NoEffect]

        if status == CutStatus.Success:
            # Shape matrix should remain positive semi-definite
            eigenvalues = np.linalg.eigvals(ell._mq)
            assert np.all(eigenvalues >= 0)

            # kappa should remain positive
            assert ell._kappa > 0

    @given(valid_ellipsoid_strategy(), valid_parallel_cut_strategy(4))
    @settings(max_examples=100)
    def test_parallel_cut_properties(self, ell_params, cut):
        """Test properties of parallel cuts."""
        kappa, xc = ell_params

        # Create 4D ellipsoid for consistent testing
        xc_4d = np.zeros(4)
        ell = Ell(kappa, xc_4d)

        grad, beta_list = cut
        beta1, beta2 = beta_list

        # Test both central and bias cut with parallel cuts
        status_central = ell.update_central_cut((grad, beta_list))

        # Reset ellipsoid
        ell = Ell(kappa, xc_4d)
        status_bias = ell.update_bias_cut((grad, beta_list))

        # Both should have valid status
        assert status_central in [
            CutStatus.Success,
            CutStatus.NoSoln,
            CutStatus.NoEffect,
        ]
        assert status_bias in [CutStatus.Success, CutStatus.NoSoln, CutStatus.NoEffect]

        if status_central == CutStatus.Success:
            # Shape matrix should remain positive semi-definite
            eigenvalues = np.linalg.eigvals(ell._mq)
            assert np.all(eigenvalues >= 0)

    @given(valid_ellipsoid_strategy())
    def test_ellipsoid_volume_monotonicity(self, ell_params):
        """Test that ellipsoid volume decreases or stays the same after cuts."""
        kappa, xc = ell_params

        # Create 3D ellipsoid for volume calculation
        xc_3d = np.zeros(3)
        ell = Ell(kappa, xc_3d)

        # Calculate initial volume (proportional to sqrt(det(kappa * M)))
        initial_volume = ell._kappa * np.sqrt(np.linalg.det(ell._mq))

        # Apply a central cut with small gradient to ensure feasibility
        grad = np.array([0.1, 0.1, 0.1])
        status = ell.update_central_cut((grad, 0.0))

        if status == CutStatus.Success:
            # Volume should decrease
            final_volume = ell._kappa * np.sqrt(np.linalg.det(ell._mq))
            assert final_volume <= initial_volume

    @given(valid_ellipsoid_strategy(), valid_cut_strategy(3))
    @settings(max_examples=50)
    def test_cut_consistency(self, ell_params, cut):
        """Test that different cut methods are consistent for equivalent cuts."""
        kappa, xc = ell_params

        # Create 3D ellipsoid
        xc_3d = np.zeros(3)
        ell1 = Ell(kappa, xc_3d)
        ell2 = Ell(kappa, xc_3d)

        grad, beta = cut

        # For beta = 0, central cut and bias cut should be equivalent
        status_central = ell1.update_central_cut((grad, 0.0))
        status_bias = ell2.update_bias_cut((grad, 0.0))

        assert status_central == status_bias

        if status_central == CutStatus.Success:
            # Results should be identical
            assert np.allclose(ell1.xc(), ell2.xc())
            assert np.allclose(ell1._mq, ell2._mq)
            assert ell1._kappa == ell2._kappa

    @given(
        st.floats(min_value=0.001, max_value=10),
        st.lists(st.floats(min_value=0.1, max_value=5), min_size=3, max_size=3),
    )
    def test_diagonal_initialization_properties(self, kappa, diag_vals):
        """Test properties of diagonal matrix initialization."""
        # Ensure diagonal values are positive for positive definiteness
        assume(np.all(np.array(diag_vals) > 0))

        xc = np.zeros(3)

        # Create ellipsoid with diagonal values
        ell = Ell(diag_vals, xc)

        # Shape matrix should be diagonal
        assert np.allclose(ell._mq, np.diag(diag_vals))

        # kappa should be 1 for diagonal initialization
        assert ell._kappa == 1.0

        # Diagonal values should be positive for positive definiteness
        eigenvalues = np.linalg.eigvals(ell._mq)
        assert np.all(eigenvalues > 0)

    @given(valid_ellipsoid_strategy(), valid_cut_strategy(3))
    @settings(max_examples=50)
    def test_tsq_calculation_property(self, ell_params, cut):
        """Test that tsq calculation follows the expected formula."""
        kappa, xc = ell_params

        # Create 3D ellipsoid for consistent testing
        xc_3d = np.zeros(3)
        ell = Ell(kappa, xc_3d)

        grad, beta = cut

        # Store initial values
        initial_kappa = ell._kappa
        initial_mq = ell._mq.copy()

        status = ell.update_bias_cut((grad, beta))

        if status == CutStatus.Success:
            # tsq should be kappa * grad^T * M * grad
            omega = grad.dot(initial_mq @ grad)
            expected_tsq = initial_kappa * omega
            assert ell.tsq() == approx(expected_tsq)

    @given(valid_ellipsoid_strategy())
    def test_no_defer_trick_property(self, ell_params):
        """Test the no_defer_trick property."""
        kappa, xc = ell_params

        # Create 3D ellipsoid for consistency
        xc_3d = np.zeros(3)
        ell = Ell(kappa, xc_3d)

        # Enable no_defer_trick
        ell.no_defer_trick = True

        # Apply a cut
        grad = np.array([1.0, 1.0, 1.0])
        status = ell.update_bias_cut((grad, 0.1))

        if status == CutStatus.Success:
            # kappa should be reset to 1.0
            assert ell._kappa == 1.0

    @given(st.lists(st.floats(min_value=0, max_value=0), min_size=2, max_size=5))
    def test_zero_gradient_error(self, grad_values):
        """Test that zero gradient raises ValueError."""
        # Create zero gradient of appropriate dimension
        ndim = len(grad_values)
        grad = np.zeros(ndim)

        ell = Ell(1.0, np.zeros(ndim))

        with pytest.raises(ValueError, match="Gradient cannot be a zero vector"):
            ell.update_central_cut((grad, 0.0))
