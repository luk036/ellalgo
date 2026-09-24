# Transpile templates (`.tpy`)

These files are **not** part of the `ellalgo` package and are **not** collected by
pytest. They are scratch templates kept for porting the numerical core to the
sibling C++ (`ellalgo-cpp`) and Rust (`ellalgo-rs`) implementations.

They were previously stored under `src/ellalgo/` and `tests/`, where they looked
like importable Python (and one, `test_quasicvx_stable.tpy`, silently never ran).
They are unimportable by design: several reference modules that do not exist here
(e.g. `from .chol_ext import LDLTMgr`) and use older APIs that have since drifted.

Do not import them. When syncing with the C++/Rust repos, treat the live `.py`
sources as the source of truth.
