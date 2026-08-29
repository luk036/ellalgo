"""
Test the LMIProblem facade and LMI oracle factory.
"""

import numpy as np

from ellalgo.ell import Ell
from ellalgo.ell_stable import EllStable
from ellalgo.lmi_problem import LMIProblem
from ellalgo.oracles.lmi_factory import (
    make_lmi0_oracle,
    make_lmi_old_oracle,
    make_lmi_oracle,
)


def _sample_problem() -> tuple[list[np.ndarray], np.ndarray]:
    """Return the standard 2x2 LMI test data (F, B)."""
    mat_f = [
        np.array([[-7.0, -11.0], [-11.0, 3.0]]),
        np.array([[7.0, -18.0], [-18.0, 8.0]]),
        np.array([[-2.0, -8.0], [-8.0, 1.0]]),
    ]
    mat_b = np.array([[33.0, -9.0], [-9.0, 26.0]])
    return mat_f, mat_b


def test_make_lmi_oracle_factory_produces_working_oracle() -> None:
    """The factory should build an oracle equivalent to direct construction."""
    mat_f, mat_b = _sample_problem()
    omega = make_lmi_oracle(mat_f, mat_b)
    x = np.array([0.0, 0.0, 0.0])
    assert omega.assess_feas(x) is None  # origin is feasible (B is PD)


def test_make_lmi_old_oracle_equivalent() -> None:
    """Lazy and explicit LMI oracles should agree on feasibility."""
    mat_f, mat_b = _sample_problem()
    x = np.array([0.0, 0.0, 0.0])
    lazy_cut = make_lmi_oracle(mat_f, mat_b).assess_feas(x)
    old_cut = make_lmi_old_oracle(mat_f, mat_b).assess_feas(x)
    assert lazy_cut is None and old_cut is None


def test_make_lmi0_oracle_factory() -> None:
    """The LMI0 factory should build a working oracle."""
    mat_f = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ]
    omega = make_lmi0_oracle(mat_f)
    assert omega.assess_feas(np.array([1.0, 0.0, 1.0])) is None
    assert omega.assess_feas(np.array([-1.0, 0.0, -1.0])) is not None


def test_lmi_problem_facade_solves_feasibility() -> None:
    """The facade should solve an LMI feasibility problem end-to-end."""
    mat_f, mat_b = _sample_problem()
    problem = LMIProblem(mat_f, mat_b)
    x, niter = problem.solve_feas(10.0, np.array([0.0, 0.0, 0.0]))
    assert x is not None
    assert niter < 2000


def test_lmi_problem_facade_radii() -> None:
    """The facade should accept per-axis radii for the initial ellipsoid."""
    mat_f, mat_b = _sample_problem()
    problem = LMIProblem(mat_f, mat_b)
    x, niter = problem.solve_feas(
        np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 0.0])
    )
    assert x is not None
    assert niter < 2000


def test_ell_from_radii_and_alpha() -> None:
    """Named constructors should match the direct constructors."""
    x = np.array([0.0, 0.0])
    radii = np.array([2.0, 3.0])
    assert np.allclose(Ell.from_radii(radii, x)._mq, Ell(radii, x)._mq)
    assert Ell.from_alpha(10.0, x)._kappa == Ell(10.0, x)._kappa
    assert np.allclose(EllStable.from_alpha(10.0, x)._mq, EllStable(10.0, x)._mq)
