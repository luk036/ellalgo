# Changelog

## Version 0.8 (2026-09-17)

### Features
- **Chebyshev-center demo + benchmark**: New `demo/chebyshev_center.py` solves the Chebyshev center problem with the ellipsoid method and cross-checks it against a CVXPY reference; `benches/bench_chebyshev.py` benchmarks CVXPY vs the ellipsoid method across problem sizes, with SVG result plots. (#7f2ad95)
- **FIR lowpass benchmark vs CVXPY**: Ports the Wu-Boyd-Vandenberghe spectral-factorization FIR lowpass design problem to a benchmark comparing CVXPY (reference LP) against `LowpassOracle` for filter lengths n = 24..80. (#a89a764)
- **Run-time comparison plots**: Plotting scripts and generated SVG figures for the min-eigenvalue LMI (EVP) and FIR lowpass experiments, comparing CVXPY vs Python vs C++ ellipsoid run times. (#a8364eb)
- **`RoundRobin.peek_next` / `RoundRobin.seek`**: New cursor accessors so the vectorized lowpass oracle can find the first violating band without advancing the round-robin cursor. (#ea14d4d)

### Bug Fixes
- **`bsearch` could never terminate early**: The stopping test `tau < options.tolerance` applies an *absolute* threshold to a scale-dependent bracket width. Once the bracket underflows at its own magnitude the test is unreachable, so `max_iters` becomes the only stopping rule and the loop spins without refining anything — on corr-solver's `lsq_corr_poly` (4 variables, `upper = 939`), `tau` reached 1.11e-16 after ~200 iterations and stayed pinned for the remaining 1800. Added a stall guard that stops when `lower + tau` is no longer strictly inside the bracket; the returned value is bit-identical and iterations drop from 2000 to 61. A 10-instance battery (varying grid, basis count and Y scale) gives identical or 1-ulp results throughout, and several instances also collapse their inner feasibility work (945,759 → 17,457 inner iterations at site=6x5 m=4). (#f51a943)

### Performance
- **`EllStable._update_core` vectorized**: Replaced the pure-Python O(n²) loops for forward substitution, D-scaling, back substitution and the rank-1 inner update with whole-slice numpy operations, leaving only the outer O(n) sweep in Python. ~2.3x faster at n=32 and ~4.5x at n=64 (a small n=2 regression from numpy slice overhead is accepted). (#b6e49cd)
- **Lowpass oracle vectorized**: The oracle scanned the spectrum row by row, calling `ndarray.dot` ~1.6M times (45% of the benchmark). Each band is now evaluated with a single BLAS gemv and the first violation is found via boolean masks. Benchmark: 14.60s → 3.73s (3.91x) parallel and 35.29s → 8.16s (4.33x) single. Also hoisted `Ell._update_core`'s `np.finfo` constant, replaced `np.all(grad == 0.0)` with `not grad.any()`, and used `grad_t[:, None] * grad_t` instead of `np.outer` (Ell micro-benchmark 1.09–1.40x). (#ea14d4d)

### Testing & Code Quality
- **`bsearch` regression test**: Added `test_bsearch_stops_at_float_resolution`, which runs 2000 iterations against an unguarded `bsearch` and fails the assertion. (#f51a943)
- **Generated-asset normalization**: Applied `pre-commit run --all-files`, which rewrote the line endings of the generated benchmark SVGs. (#482b24a)

### Documentation
- **README badge cleanup**: Removed the stale, commented-out Cirrus CI / Coveralls badge block, superseded by the current PyScaffold, ReadTheDocs and codecov badges. (#0580c18)

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
