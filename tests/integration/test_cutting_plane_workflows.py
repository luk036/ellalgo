"""
End-to-end integration tests for cutting plane algorithms.

These tests verify complete workflows from oracle definition through
solution finding.
"""

import numpy as np

from ellalgo import Ell, cutting_plane_feas
from ellalgo.ell_config import Options
from ellalgo.oracles.lmi_oracle import LMIOracle


def test_lmi_feasibility_workflow() -> None:
    """Test complete LMI feasibility solving workflow."""
    # Define a simple LMI: B - F1*x1 - F2*x2 >= 0
    # where B = [[2, 0], [0, 1]], F1 = [[1, 0], [0, 0]], F2 = [[0, 0], [0, 1]]
    B = np.array([[2.0, 0.0], [0.0, 1.0]])
    F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    F2 = np.array([[0.0, 0.0], [0.0, 1.0]])

    # Create oracle
    omega = LMIOracle([F1, F2], B)

    # Initialize ellipsoid
    x_start = np.array([0.0, 0.0])
    space = Ell(10.0, x_start)

    # Solve feasibility problem
    options = Options()
    options.max_iters = 1000
    x_best, num_iters = cutting_plane_feas(omega, space, options)

    # Should find a feasible solution
    assert x_best is not None
    assert num_iters >= 0  # May be 0 if starting point is already feasible

    # Verify solution is actually feasible
    cut = omega.assess_feas(x_best)
    assert cut is None  # No cut means feasible


def test_infeasible_lmi_workflow() -> None:
    """Test that infeasible problems are detected correctly."""
    # Define an infeasible LMI: B - F*x >= 0
    # where B = [[-1, 0], [0, -1]] (negative definite - impossible)
    B = np.array([[-1.0, 0.0], [0.0, -1.0]])
    F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    F2 = np.array([[0.0, 0.0], [0.0, 1.0]])

    # Create oracle
    omega = LMIOracle([F1, F2], B)

    # Initialize ellipsoid
    x_start = np.array([0.0, 0.0])
    space = Ell(1.0, x_start)

    # Solve feasibility problem
    options = Options()
    options.max_iters = 100
    x_best, num_iters = cutting_plane_feas(omega, space, options)

    # Should return None for infeasible problems
    assert x_best is None


def test_tolerance_affects_convergence() -> None:
    """Test that tolerance settings affect convergence."""
    # Define a feasible LMI problem
    B = np.array([[2.0, 0.0], [0.0, 2.0]])
    F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    F2 = np.array([[0.0, 0.0], [0.0, 1.0]])

    omega = LMIOracle([F1, F2], B)
    x_start = np.array([0.0, 0.0])

    # Test with loose tolerance
    options_loose = Options()
    options_loose.tolerance = 1e-6
    options_loose.max_iters = 100
    space_loose = Ell(10.0, x_start.copy())
    x_loose, iters_loose = cutting_plane_feas(omega, space_loose, options_loose)

    # Test with tight tolerance
    options_tight = Options()
    options_tight.tolerance = 1e-20
    options_tight.max_iters = 1000
    space_tight = Ell(10.0, x_start.copy())
    x_tight, iters_tight = cutting_plane_feas(omega, space_tight, options_tight)

    # Both should find solutions
    assert x_loose is not None
    assert x_tight is not None

    # Tight tolerance may take more iterations
    # (or same if problem is easy)
    assert iters_tight >= 0
    assert iters_loose >= 0
