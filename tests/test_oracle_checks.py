import numpy as np
import pytest

from ellalgo.oracle_checks import (
    check_cut_valid,
    check_lmi_witness,
    check_subgradient,
    verify_cut_valid,
    verify_lmi_witness,
    verify_subgradient,
)


def test_cut_valid_single() -> None:
    x0 = np.zeros(2)
    cut = (np.array([1.0, 1.0]), 0.0)
    feasible = [np.array([-1.0, 0.0]), np.array([0.0, -2.0]), np.array([-1.0, -1.0])]
    assert check_cut_valid(cut, x0, feasible)


def test_cut_invalid_single_detected() -> None:
    x0 = np.zeros(2)
    cut = (np.array([1.0, 1.0]), -0.5)
    feasible = [np.array([2.0, 0.0])]
    assert not check_cut_valid(cut, x0, feasible)
    with pytest.raises(AssertionError):
        verify_cut_valid(cut, x0, feasible)


def test_cut_valid_parallel() -> None:
    x0 = np.zeros(2)
    cut = (np.array([1.0, 0.0]), (-1.0, 0.0))
    feasible = [np.array([0.0, 5.0]), np.array([1.0, -3.0])]
    assert check_cut_valid(cut, x0, feasible)
    assert not check_cut_valid(cut, x0, [np.array([2.0, 0.0])])


def test_cut_valid_empty_feasible_set() -> None:
    assert check_cut_valid((np.array([1.0]), 0.0), np.zeros(1), [])


def test_subgradient_correct() -> None:
    func = lambda x: float(x @ x)  # noqa: E731
    x = np.array([1.0, -2.0])
    assert check_subgradient(func, x, 2.0 * x)


def test_subgradient_wrong_detected() -> None:
    func = lambda x: float(x @ x)  # noqa: E731
    x = np.array([1.0, -2.0])
    assert not check_subgradient(func, x, np.zeros(2), step=0.1)
    with pytest.raises(AssertionError):
        verify_subgradient(func, x, np.zeros(2), step=0.1)


def test_lmi_witness() -> None:
    v = np.array([1.0, 0.0])
    assert check_lmi_witness(-np.eye(2), v)
    assert not check_lmi_witness(np.eye(2), v)
    with pytest.raises(AssertionError):
        verify_lmi_witness(np.eye(2), v)
