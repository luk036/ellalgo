# Changelog

## Version 0.7 (2026-08-31)

### Features
- **`LMIProblem` facade + LMI oracle factory**: New `LMIProblem` class owns the F/B matrices and drives `cutting_plane_feas` in a single call, plus `make_lmi_oracle` / `make_lmi0_oracle` / `make_lmi_old_oracle` factory entry points. Added `EllBase.from_radii` / `from_alpha` named constructors. Exported `LMIProblem` in `__init__`. (#8021f31)

### Bug Fixes
- **Python 3.9 import crash**: `round_robin` and `lmi_problem` used PEP 604 annotations (`X | None`, `float | np.ndarray`) evaluated at runtime on Python 3.9; added `from __future__ import annotations` to both, matching the `ell_typing` convention. (#d3735b7)
- **Config dedup**: Removed duplicate `mypy.ini` config and fixed the `ArrayType` annotation. (#ba0116f)
- **RTD build fix**: Added `matplotlib` and `numpy` to `docs/requirements.txt` so ReadTheDocs can build the docs. (#454d7b5)

### Code Cleanup
- **Shared `EllBase` (Strategy)**: Extracted the duplicated public API (constructor, `xc`/`set_xc`/`tsq`, `update_*` wrappers) of `Ell`/`EllStable` into `EllBase` with a `_update_core` Template-Method hook; each strategy overrides only its update core. (#1c305a1)
- **Shared `LMIBase` (Template Method)**: Collapsed the identical `assess_feas` skeleton (factor → witness → sym_quad → pack cut) of `LMIOracle`/`LMI0Oracle`/`LMIOldOracle` into `LMIBase._assess`; subclasses supply only their `get_elem` closure and sign. (#5b6ac05)
- **`RoundRobin` helper (Strategy)**: Extracted the duplicated `idx += 1; if idx == N: idx = 0` idiom; `ProfitOracle` keeps a readable `idx`, `LowpassOracle` keeps public `idx1`/`idx2`/`idx3` mirrors. (#b9d0ae4)
- **`OptimQState` state machine**: Encapsulated the scattered `x_best` + retry bookkeeping of the discrete-optimization loop with `on_shrunk`/`on_update` transitions (exact return semantics preserved). (#fddc4c1)
- **`LDLTMgr` factor skeleton**: Unified `factor` and `factor_with_allow_semidefinite` into a shared `_factor_impl(get_elem, allow_semidefinite)` differing only in pivot policy; public signatures and doctests unchanged. (#ba66b48)
- **Docstring de-slop**: Removed AI-slop boilerplate from docstrings and comments. (#b3c4839)
- **Formatting**: Applied black to `conjugate_gradient`. (#55b88dc)

### Maintenance
- **Config cleanup**: Migrated flake8 config to `.flake8` and fixed project URLs in metadata. (#f795342)

### Build & CI
- **CI cleanup**: Removed the stale `.bak` workflow file. (#56de147)

## Version 0.6 (2026-07-16)

### Features
- **Pre-allocated scratch buffers for `EllStable`**: `_update_core` now uses 3 pre-allocated numpy arrays (`_inv_lower_g`, `_inv_diag_inv_lower_g`, `_g_t`), eliminating all 5 per-call allocations from the hot path. Omega is computed inline (no `gg_t` buffer). Achieves ~5.4x speedup matching Rust's scratch buffer strategy. (#cc9f929)
- **`SingleCut` type annotation**: New `SingleCut` type alias introduced in `ell_typing` and used consistently across `ell`, `ell_stable`, and `cutting_plane` modules, improving type clarity for single vs. parallel cuts. (#cc9f929)
- **Dual-purpose `_g_t` buffer**: The `_g_t` buffer is reused as working vector `v` for the rank-1 LDL^T update after back-substitution (Rust-style dual-purpose pattern). (#cc9f929)

### Bug Fixes
- **Denormal omega overflow guard**: Added threshold check in `Ell.update()` to catch denormal `omega` values that would cause `sigma/omega` overflow in the rank-1 matrix update. (#af17594)
- **CI repair**: Fixed broken `entry_points` configuration and remaining `skeleton` imports that were breaking the CI pipeline. (#ecec2dd)
- **macOS CI stability**: Added `continue-on-error` to the Coveralls step and updated GitHub Actions versions to fix macOS CI failures. (#7cb1dd0)

### Performance
- **~5.4x speedup in `EllStable._update_core`**: Pre-allocated scratch buffers eliminate all per-call numpy array allocations from the critical path. Performance now matches the Rust implementation's allocation strategy. (#cc9f929)

### Testing & Code Quality
- **Test deduplication + coverage**: Removed redundant tests and added targeted coverage tests, raising coverage from 92% → 98%. (#5ae382e)
- **Removed dead test files**: Deleted `test_skeleton.py`, `test_ell_more.py`, `test_profit.py` (redundant or superseded). (#5ae382e)
- **New test suites**: Added `test_ell_typing.py`, `test_spectral_fact.py`, `test_conjugate_gradient.py`, and expanded `test_cutting_plane.py`. (#5ae382e)
- **Pre-commit cleanup**: Applied `pre-commit run --all-files` for consistent formatting across the codebase. (#8219ab5)

### Code Cleanup
- **Removed PyScaffold boilerplate**: Deleted `skeleton.py` (unused Fibonacci CLI scaffold) and its associated `test_skeleton.py`. (#c94c990)
- **Dropped Python < 3.9 compat**: Removed `importlib-metadata` conditional dependency and compat guard from `__init__.py` (project now targets 3.10+). (#c94c990)
- **Config cleanup**: Removed dead `fibonacci = skeleton:run` entry point comment from `setup.cfg` and unused `ignore_missing_imports` from `mypy.ini`. (#c94c990)
- **Deduplicated LICENSE**: Removed duplicate `LICENSE` file. (#c94c990)
- **Updated `.gitignore`**: Added `.ruff_cache/` and `.benchmarks/` directories. (#c94c990, #2b37122)

### Documentation
- **Polyglot performance comparison report**: Added `compare-py-cpp-rs.md` — a detailed benchmark comparison of ellipsoid method implementations across Python, C++, and Rust, with analysis of the performance gap and strategies to close it. (#133e853)
- **Docstring improvements**: Enhanced module-level docstrings across `ell_stable.py` and related files. (#cc9f929)
