import numpy as np

from ellalgo import CutStatus, Ell, EllStable, Options, SolverStatus, cutting_plane_feas
from ellalgo.cutting_plane import _stalled


def test_stalled_helper() -> None:
    assert not _stalled(1.0, None)
    assert not _stalled(0.0, 1.0)
    assert _stalled(1.0, 1.0)
    assert _stalled(1.0, 1.0 + 1e-18)
    assert not _stalled(1.0, 1.5)


def test_invariants_hold() -> None:
    ell = Ell(1.0, np.zeros(2))
    assert ell._invariants_hold(0.5, 1.1)
    assert not ell._invariants_hold(1.5, 1.1)
    assert not ell._invariants_hold(0.5, -0.1)
    assert not ell._invariants_hold(float("nan"), 0.5)


def test_validate_pd_direct() -> None:
    ell = Ell(1.0, np.zeros(3))
    assert ell._validate_pd()
    ell._mq = np.diag([1.0, -1.0, 1.0])
    assert not ell._validate_pd()


def test_validate_pd_stable() -> None:
    ell = EllStable(1.0, np.zeros(3))
    assert ell._validate_pd()
    ell._mq[1, 1] = -1.0
    assert not ell._validate_pd()


class _AlwaysCut:
    def assess_feas(self, xc: np.ndarray):
        return (np.array([1.0, 1.0]), 0.0)


class _UnknownSpace(Ell):
    def update_bias_cut(self, cut) -> CutStatus:
        return CutStatus.Unknown


class _StuckSpace(Ell):
    def tsq(self) -> float:
        return 1e-3

    def update_bias_cut(self, cut) -> CutStatus:
        return CutStatus.Success


def test_numerical_breakdown_reports_failed() -> None:
    res = cutting_plane_feas(_AlwaysCut(), _UnknownSpace(1.0, np.zeros(2)))
    assert res.status == SolverStatus.Failed


def test_stall_guard_reports_stalled() -> None:
    res = cutting_plane_feas(_AlwaysCut(), _StuckSpace(1.0, np.zeros(2)))
    assert res.status == SolverStatus.Stalled


def test_stall_guard_can_be_disabled() -> None:
    opts = Options(tolerance=1e-20, stall_guard=False, max_iters=5)
    res = cutting_plane_feas(_AlwaysCut(), _StuckSpace(1.0, np.zeros(2)), opts)
    assert res.status == SolverStatus.MaxIters
