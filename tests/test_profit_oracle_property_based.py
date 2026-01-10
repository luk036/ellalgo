"""
Property-based tests for the ProfitOracle classes using Hypothesis.

These tests verify that the profit oracle methods satisfy mathematical properties
that should hold regardless of the specific input values.
"""

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pytest import approx

from ellalgo.oracles.profit_oracle import ProfitOracle, ProfitQOracle, ProfitRbOracle


class TestProfitOracleProperties:
    """Property-based tests for ProfitOracle mathematical properties."""

    @st.composite
    def valid_profit_params(draw):
        """Generate valid profit oracle parameters."""
        # Unit price (positive)
        unit_price = draw(st.floats(min_value=0.1, max_value=100.0))
        # Scale factor (positive)
        scale = draw(st.floats(min_value=0.1, max_value=10.0))
        # Limit (positive)
        limit = draw(st.floats(min_value=1.0, max_value=100.0))

        return (unit_price, scale, limit)

    @st.composite
    def valid_elasticities(draw):
        """Generate valid elasticity parameters."""
        # Elasticities should be positive and sum to less than 1 for diminishing returns
        alpha = draw(st.floats(min_value=0.01, max_value=0.8))
        beta = draw(st.floats(min_value=0.01, max_value=0.8))

        # Ensure sum < 1 for economic feasibility
        assume(alpha + beta < 0.99)

        return np.array([alpha, beta])

    @st.composite
    def valid_price_out(draw):
        """Generate valid output price parameters."""
        v1 = draw(st.floats(min_value=0.1, max_value=10.0))
        v2 = draw(st.floats(min_value=0.1, max_value=10.0))

        return np.array([v1, v2])

    @st.composite
    def valid_solution_point(draw):
        """Generate valid solution points in log-space."""
        # Log-space values (can be negative)
        x1 = draw(st.floats(min_value=-3.0, max_value=3.0))
        x2 = draw(st.floats(min_value=-3.0, max_value=3.0))

        return np.array([x1, x2])

    @given(valid_profit_params(), valid_elasticities(), valid_price_out())
    def test_initialization_properties(self, params, elasticities, price_out):
        """Test that ProfitOracle initialization preserves basic properties."""
        oracle = ProfitOracle(params, elasticities, price_out)

        # Parameters should be preserved
        assert oracle.elasticities == approx(elasticities)
        assert oracle.price_out == approx(price_out)

        # Precomputed values should be correct
        unit_price, scale, limit = params
        expected_log_pA = math.log(unit_price * scale)
        expected_log_k = math.log(limit)

        assert oracle.log_pA == approx(expected_log_pA)
        assert oracle.log_k == approx(expected_log_k)

        # Constraint functions should be available
        assert len(oracle.fns) == 2
        assert len(oracle.grads) == 2

    @given(
        valid_profit_params(),
        valid_elasticities(),
        valid_price_out(),
        valid_solution_point(),
    )
    @settings(max_examples=100)
    def test_constraint_fn1_properties(self, params, elasticities, price_out, x):
        """Test properties of the first constraint function."""
        oracle = ProfitOracle(params, elasticities, price_out)

        # fn1(x) = x[0] - log(k)
        result = oracle.fn1(x, 0.0)  # gamma not used in fn1

        expected = x[0] - oracle.log_k
        assert result == approx(expected)

        # Gradient should be [1, 0]
        grad = oracle.grad1(0.0)
        expected_grad = np.array([1.0, 0.0])
        assert grad == approx(expected_grad)

    @given(
        valid_profit_params(),
        valid_elasticities(),
        valid_price_out(),
        valid_solution_point(),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=100)
    def test_constraint_fn2_properties(self, params, elasticities, price_out, x, gamma):
        """Test properties of the second constraint function."""
        oracle = ProfitOracle(params, elasticities, price_out)

        # Call fn2 to set up intermediate values
        result = oracle.fn2(x, gamma)

        # Expected: log(gamma + vy) - log_Cobb
        # where log_Cobb = log(pA) + alpha*log(y1) + beta*log(y2)
        # and vy = v1*y1 + v2*y2
        # with y1 = exp(x[0]), y2 = exp(x[1])
        y1, y2 = np.exp(x)
        v1, v2 = price_out
        alpha, beta = elasticities

        expected_log_Cobb = oracle.log_pA + alpha * x[0] + beta * x[1]
        expected_vy = v1 * y1 + v2 * y2
        expected_result = math.log(gamma + expected_vy) - expected_log_Cobb

        assert result == approx(expected_result)
        assert oracle.log_Cobb == approx(expected_log_Cobb)
        assert oracle.vy == approx(expected_vy)

        # Check intermediate q values
        expected_q = np.array([v1 * y1, v2 * y2])
        assert oracle.q == approx(expected_q)

    @given(
        valid_profit_params(),
        valid_elasticities(),
        valid_price_out(),
        valid_solution_point(),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=100)
    def test_gradient_fn2_properties(self, params, elasticities, price_out, x, gamma):
        """Test properties of the second constraint gradient."""
        oracle = ProfitOracle(params, elasticities, price_out)

        # Call fn2 first to set up intermediate values
        oracle.fn2(x, gamma)

        grad = oracle.grad2(gamma)

        # Expected: q/(gamma + vy) - elasticities
        expected_grad = oracle.q / (gamma + oracle.vy) - elasticities

        assert grad == approx(expected_grad)

    @given(
        valid_profit_params(),
        valid_elasticities(),
        valid_price_out(),
        valid_solution_point(),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50)
    def test_feasibility_assessment_properties(
        self, params, elasticities, price_out, x, gamma
    ):
        """Test properties of feasibility assessment."""
        oracle = ProfitOracle(params, elasticities, price_out)

        cut = oracle.assess_feas(x, gamma)

        # If cut is returned, it should be valid
        if cut is not None:
            grad, beta = cut

            # Gradient should have correct dimension
            assert len(grad) == 2

            # Beta should be positive (constraint violation)
            assert beta > 0

            # Round-robin index should be updated
            assert oracle.idx in [0, 1]

    @given(
        valid_profit_params(),
        valid_elasticities(),
        valid_price_out(),
        valid_solution_point(),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50)
    def test_optimality_assessment_properties(
        self, params, elasticities, price_out, x, gamma
    ):
        """Test properties of optimality assessment."""
        oracle = ProfitOracle(params, elasticities, price_out)

        cut, gamma_new = oracle.assess_optim(x, gamma)

        # Cut should always be returned
        assert cut is not None
        grad, beta = cut

        # Gradient should have correct dimension
        assert len(grad) == 2

        # If feasible, beta should be 0 (optimality cut)
        if gamma_new is not None:
            assert beta == 0.0
            # New gamma should be positive (profit)
            assert gamma_new >= 0.0

    @given(valid_profit_params(), valid_elasticities(), valid_price_out())
    def test_economic_feasibility_properties(self, params, elasticities, price_out):
        """Test economic feasibility properties."""
        oracle = ProfitOracle(params, elasticities, price_out)

        # Create a feasible point (well within constraints)
        unit_price, scale, limit = params
        feasible_x = np.array(
            [math.log(limit * 0.5), math.log(1.0)]
        )  # y1 = 0.5*limit, y2 = 1.0

        # Should be feasible for some gamma
        gamma = 1.0
        oracle.assess_feas(feasible_x, gamma)

        # May or may not be feasible depending on parameters
        # But the assessment should not crash

        # Optimality assessment should work
        cut_opt, gamma_new = oracle.assess_optim(feasible_x, gamma)
        assert cut_opt is not None


class TestProfitRbOracleProperties:
    """Property-based tests for ProfitRbOracle mathematical properties."""

    @st.composite
    def valid_robust_params(draw):
        """Generate valid robust oracle parameters."""
        # Base parameters
        unit_price = draw(st.floats(min_value=1.0, max_value=100.0))
        scale = draw(st.floats(min_value=0.1, max_value=10.0))
        limit = draw(st.floats(min_value=1.0, max_value=100.0))

        # Uncertainty parameters (should be small relative to base values)
        e1 = draw(st.floats(min_value=0.01, max_value=0.1))  # Elasticity uncertainty
        e2 = draw(st.floats(min_value=0.01, max_value=0.1))
        e3 = draw(
            st.floats(min_value=0.01, max_value=unit_price * 0.1)
        )  # Price uncertainty
        e4 = draw(st.floats(min_value=0.01, max_value=limit * 0.1))  # Limit uncertainty
        e5 = draw(st.floats(min_value=0.01, max_value=0.1))  # Input price uncertainty

        return (unit_price, scale, limit), (e1, e2, e3, e4, e5)

    @given(
        valid_robust_params(),
        TestProfitOracleProperties.valid_elasticities(),
        TestProfitOracleProperties.valid_price_out(),
    )
    def test_robust_initialization_properties(
        self, robust_params, elasticities, price_out
    ):
        """Test that ProfitRbOracle initialization preserves robust properties."""
        params, vparams = robust_params
        oracle = ProfitRbOracle(params, elasticities, price_out, vparams)

        # Should have underlying omega oracle
        assert oracle.omega is not None
        assert isinstance(oracle.omega, ProfitOracle)

        # Uncertainty parameters should be stored
        assert oracle.uie == [vparams[0], vparams[1]]  # e1, e2

        # Base elasticities should be preserved
        assert np.allclose(oracle.elasticities, elasticities)

    @given(
        valid_robust_params(),
        TestProfitOracleProperties.valid_elasticities(),
        TestProfitOracleProperties.valid_price_out(),
        TestProfitOracleProperties.valid_solution_point(),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50)
    def test_robust_adjustment_properties(
        self, robust_params, elasticities, price_out, x, gamma
    ):
        """Test properties of robust parameter adjustments."""
        params, vparams = robust_params
        oracle = ProfitRbOracle(params, elasticities, price_out, vparams)

        cut, gamma_new = oracle.assess_optim(x, gamma)

        # Should return a valid cut
        assert cut is not None
        grad, beta = cut

        # Gradient should have correct dimension
        assert len(grad) == 2

        # Robust adjustments should be applied to omega elasticities
        adjusted_elasticities = oracle.omega.elasticities

        # Adjustments should be in the expected direction
        for i in [0, 1]:
            if x[i] > 0:
                # Should decrease elasticity for positive x
                assert adjusted_elasticities[i] <= elasticities[i] + vparams[i]
            else:
                # Should increase elasticity for non-positive x
                assert adjusted_elasticities[i] >= elasticities[i] - vparams[i]


class TestProfitQOracleProperties:
    """Property-based tests for ProfitQOracle mathematical properties."""

    @given(
        TestProfitOracleProperties.valid_profit_params(),
        TestProfitOracleProperties.valid_elasticities(),
        TestProfitOracleProperties.valid_price_out(),
    )
    def test_discrete_initialization_properties(self, params, elasticities, price_out):
        """Test that ProfitQOracle initialization preserves discrete properties."""
        oracle = ProfitQOracle(params, elasticities, price_out)

        # Should have underlying omega oracle
        assert oracle.omega is not None
        assert isinstance(oracle.omega, ProfitOracle)

        # Initial discrete solution should be zeros
        assert np.allclose(oracle.xd, np.array([0.0, 0.0]))

    @given(
        TestProfitOracleProperties.valid_profit_params(),
        TestProfitOracleProperties.valid_elasticities(),
        TestProfitOracleProperties.valid_price_out(),
        TestProfitOracleProperties.valid_solution_point(),
        st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50)
    def test_discrete_rounding_properties(
        self, params, elasticities, price_out, x, gamma
    ):
        """Test properties of discrete rounding mechanism."""
        oracle = ProfitQOracle(params, elasticities, price_out)

        # First assessment (retry=False)
        cut, eval_point, gamma_new, retry_flag = oracle.assess_optim_q(x, gamma, False)

        # Should return valid cut
        assert cut is not None
        grad, beta = cut

        # First try should evaluate at a valid point
        assert len(eval_point) == len(x)

        # If feasible, should set discrete solution
        if gamma_new is not None:
            # Check that discrete solution is properly rounded
            yd = np.round(np.exp(x))
            yd[yd == 0] = 1.0  # Zero protection
            expected_xd = np.log(yd)
            assert np.allclose(oracle.xd, expected_xd)

        # Second assessment (retry=True) if needed
        if retry_flag:
            cut2, eval_point2, gamma_new2, retry_flag2 = oracle.assess_optim_q(
                x, gamma, True
            )

            # Second try should evaluate at discrete point
            assert np.allclose(eval_point2, oracle.xd)

            # Should not request another retry
            assert not retry_flag2

    @given(
        TestProfitOracleProperties.valid_profit_params(),
        TestProfitOracleProperties.valid_elasticities(),
        TestProfitOracleProperties.valid_price_out(),
    )
    def test_integer_solution_properties(self, params, elasticities, price_out):
        """Test properties of integer solutions."""
        oracle = ProfitQOracle(params, elasticities, price_out)

        # Test with a point that rounds to positive integers
        x = np.array([math.log(2.5), math.log(3.7)])  # Rounds to [3, 4]
        gamma = 1.0

        # First assessment
        cut, eval_point, gamma_new, retry_flag = oracle.assess_optim_q(x, gamma, False)

        # Check that oracle is working
        assert cut is not None
        assert len(eval_point) == len(x)

        if gamma_new is not None:
            # Check that discrete solution corresponds to integer inputs
            discrete_y = np.exp(oracle.xd)
            assert np.all(discrete_y == np.round(discrete_y))
            assert np.all(discrete_y >= 1.0)  # No zeros allowed

            # Discrete solution should correspond to integer inputs
            discrete_y = np.exp(oracle.xd)
            assert np.all(discrete_y == np.round(discrete_y))
            assert np.all(discrete_y >= 1.0)  # No zeros allowed
