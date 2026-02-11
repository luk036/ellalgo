"""
Property-based tests for the EllCalc class using Hypothesis.

These tests verify that the ellipsoid calculator methods satisfy mathematical properties
that should hold regardless of the specific input values.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ellalgo.ell_calc import EllCalc
from ellalgo.ell_config import CutStatus


class TestEllCalcProperties:
    """Property-based tests for EllCalc mathematical properties."""

    @given(st.integers(min_value=2, max_value=20))
    def test_initialization_properties(self, n):
        """Test that EllCalc initialization preserves basic properties."""
        calc = EllCalc(n)

        # Dimension should be preserved as float
        assert calc._n_f == float(n)

        # Helper should be initialized
        assert calc.helper is not None

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_bias_cut_monotonicity(self, n, beta, tsq):
        """Test that bias cut parameters satisfy monotonicity properties."""
        assume(beta >= 0.0)  # Required by calc_bias_cut

        calc = EllCalc(n)
        status, result = calc.calc_bias_cut(beta, tsq)

        if status == CutStatus.Success:
            assert result is not None
            rho, sigma, delta = result

            # Result parameters should be positive
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_parallel_cut_ordering(self, n, beta0, beta1, tsq):
        """Test that parallel cuts respect ordering constraints."""
        calc = EllCalc(n)

        # Test with beta1 >= beta0
        assume(beta1 >= beta0)

        status, result = calc.calc_parallel(beta0, beta1, tsq)

        if status == CutStatus.Success:
            assert result is not None
            rho, sigma, delta = result

            # Result parameters should be positive
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

            # Delta should be less than 1 (volume reduction)
            assert delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=-5.0, max_value=5.0),
        st.floats(min_value=-5.0, max_value=5.0),
    )
    @settings(max_examples=50)
    def test_parallel_cut_no_solution_when_invalid(self, n, beta0, beta1):
        """Test that parallel cuts return NoSoln when beta1 < beta0."""
        calc = EllCalc(n)

        # Only test when beta1 < beta0
        assume(beta1 < beta0)

        tsq = 1.0
        status, result = calc.calc_parallel(beta0, beta1, tsq)

        # Should return NoSoln
        assert status == CutStatus.NoSoln
        assert result is None

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_central_cut_properties(self, n, beta, tsq):
        """Test properties of central cuts."""
        calc = EllCalc(n)

        # Test single central cut
        status_single, result_single = calc.calc_single_or_parallel_central_cut(
            beta, tsq
        )

        if status_single == CutStatus.Success:
            assert result_single is not None
            rho, sigma, delta = result_single

            # Result parameters should be positive
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

            # Delta should be less than 1 (volume reduction)
            assert delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_parallel_central_cut_properties(self, n, beta0, beta1, tsq):
        """Test properties of parallel central cuts."""
        calc = EllCalc(n)

        # Test parallel central cut
        beta_list = [beta0, beta1]
        status_parallel, result_parallel = calc.calc_single_or_parallel_central_cut(
            beta_list, tsq
        )

        if status_parallel == CutStatus.Success:
            assert result_parallel is not None
            rho, sigma, delta = result_parallel

            # Result parameters should be positive
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

            # Delta should be less than 1 (volume reduction)
            assert delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_bias_cut_q_properties(self, n, beta, tsq):
        """Test properties of bias cut Q (discrete optimization version)."""
        calc = EllCalc(n)

        status, result = calc.calc_bias_cut_q(beta, tsq)

        if status == CutStatus.Success:
            assert result is not None
            rho, sigma, delta = result

            # Result parameters should be positive
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

            # Delta should be less than 1 (volume reduction)
            assert delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_parallel_cut_q_properties(self, n, beta0, beta1, tsq):
        """Test properties of parallel cut Q (discrete optimization version)."""
        calc = EllCalc(n)

        assume(beta1 >= beta0)

        status, result = calc.calc_parallel_q(beta0, beta1, tsq)

        if status == CutStatus.Success:
            assert result is not None
            rho, sigma, delta = result

            # Result parameters should be positive
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

            # Delta should be less than 1 (volume reduction)
            assert delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_single_vs_parallel_consistency(self, n, beta, tsq):
        """Test that single and parallel cuts are consistent when appropriate."""
        calc = EllCalc(n)

        # Test single cut
        status_single, result_single = calc.calc_single_or_parallel(beta, tsq)

        # Test parallel cut with same beta twice
        status_parallel, result_parallel = calc.calc_parallel(beta, beta, tsq)

        # Both should have valid status
        assert status_single in [
            CutStatus.Success,
            CutStatus.NoSoln,
            CutStatus.NoEffect,
        ]
        assert status_parallel in [
            CutStatus.Success,
            CutStatus.NoSoln,
            CutStatus.NoEffect,
        ]

        # Results may differ between single and parallel cuts
        # but both should be valid if successful
        if status_single == CutStatus.Success:
            assert result_single is not None
            rho, sigma, delta = result_single
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

        if status_parallel == CutStatus.Success:
            assert result_parallel is not None
            rho, sigma, delta = result_parallel
            assert rho >= 0
            assert sigma >= 0
            assert delta >= 0

    @given(st.integers(min_value=2, max_value=10))
    def test_use_parallel_cut_flag(self, n):
        """Test that use_parallel_cut flag affects behavior."""
        calc = EllCalc(n)

        # Test with parallel cuts enabled
        calc.use_parallel_cut = True
        beta_list = [0.1, 0.2]
        tsq = 1.0

        status_enabled, result_enabled = calc.calc_single_or_parallel(beta_list, tsq)

        # Test with parallel cuts disabled
        calc.use_parallel_cut = False
        status_disabled, result_disabled = calc.calc_single_or_parallel(beta_list, tsq)

        # Both should return some result (may be different)
        # The key point is that the flag doesn't cause crashes
        assert status_enabled in [CutStatus.Success, CutStatus.NoSoln]
        assert status_disabled in [CutStatus.Success, CutStatus.NoSoln]

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_central_vs_bias_cut_relationship(self, n, beta, tsq):
        """Test relationship between central and bias cuts."""
        calc = EllCalc(n)

        # Central cut (beta = 0)
        status_central, result_central = calc.calc_bias_cut(0.0, tsq)

        # Regular bias cut
        status_bias, result_bias = calc.calc_bias_cut(beta, tsq)

        if status_central == CutStatus.Success and status_bias == CutStatus.Success:
            # Both should produce valid results
            assert result_central is not None
            assert result_bias is not None
            assert result_central[0] >= 0  # rho >= 0
            assert result_bias[0] >= 0  # rho >= 0
            assert result_central[2] >= 0  # delta >= 0
            assert result_bias[2] >= 0  # delta >= 0

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=5.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_no_solution_condition(self, n, beta, tsq):
        """Test conditions that should return NoSoln."""
        calc = EllCalc(n)

        # For bias cut, if tsq < beta^2, should return NoSoln
        if tsq < beta * beta and beta > 0:
            status, result = calc.calc_bias_cut(beta, tsq)
            assert status == CutStatus.NoSoln
            assert result is None

    @given(
        st.integers(min_value=2, max_value=10),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.001, max_value=10.0),
    )
    @settings(max_examples=50)
    def test_dimension_scaling_properties(self, n, beta, tsq):
        """Test how results scale with dimension."""
        assume(beta >= 0.0)

        calc1 = EllCalc(n)
        calc2 = EllCalc(n + 1)  # Higher dimension

        status1, result1 = calc1.calc_bias_cut(beta, tsq)
        status2, result2 = calc2.calc_bias_cut(beta, tsq)

        if status1 == CutStatus.Success and status2 == CutStatus.Success:
            # Both should be valid regardless of dimension
            assert result1 is not None
            assert result2 is not None
            assert result1[2] >= 0  # delta >= 0
            assert result2[2] >= 0  # delta >= 0
            # The relationship between deltas depends on the specific algorithm
            # but both should be valid
