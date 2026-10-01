# Tunable Parameters in `ellalgo`

A complete reference of every user-tunable parameter of the algorithms in this
project, together with its default value and source location.

The library has a single dedicated configuration object (`Options`). Nearly
everything else is either problem data (oracle matrices, prices, filter specs)
or a hardcoded numerical constant. This report separates those categories
explicitly so it is clear what a caller can actually adjust.

---

## 1. Primary tuning surface — `Options`

Defined in `src/ellalgo/ell_config.py:48-60`. It is the only `@dataclass` in the
project.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `max_iters` | `int` | `2000` | Maximum iterations before returning `SolverStatus.MaxIters`. |
| `tolerance` | `float` | `1e-20` | Convergence threshold; a driver stops when `tsq() < tolerance`. |
| `verbose` | `bool` | `False` | Documented as progress output, but never read anywhere in the codebase (currently inert). |

Example:

```python
from ellalgo import Options

options = Options(max_iters=5000, tolerance=1e-10)
```

---

## 2. Algorithm entry points

Every cutting-plane / binary-search driver accepts an optional `options`
argument. When it is `None`, the driver constructs an internal `Options()`.

| Function | Signature | Location |
| --- | --- | --- |
| `cutting_plane_feas` | `(omega, space, options=None)` | `src/ellalgo/cutting_plane.py:50` |
| `cutting_plane_optim` | `(omega, space, gamma, options=None)` | `src/ellalgo/cutting_plane.py:143` |
| `cutting_plane_optim_q` | `(omega, space_q, gamma, options=None)` | `src/ellalgo/cutting_plane.py:276` |
| `bsearch` | `(omega, intrvl, options=None)` | `src/ellalgo/cutting_plane.py:332` |
| `BSearchAdaptor.__init__` | `(omega, space, options=None)` | `src/ellalgo/cutting_plane.py:394` |

`gamma` (initial objective value) and `intrvl` (lower/upper bracket) are problem
inputs rather than algorithm tunables. `BSearchAdaptor` forwards its `options`
to each feasibility subproblem it solves.

---

## 3. Search-space (ellipsoid) construction parameters

| Parameter | Default | Location | Notes |
| --- | --- | --- | --- |
| `val` | required | `src/ellalgo/ell_base.py:47` | Scalar (`kappa`) creates an identity shape matrix; an array creates a diagonal matrix with `_kappa = 1.0`. |
| `x_center` | required | `src/ellalgo/ell_base.py:47` | Initial center point. |
| `no_defer_trick` | `False` | `src/ellalgo/ell_base.py:47-52` | When `True`, the shape matrix is rescaled by `kappa` on every update. |
| `EllCalc.__init__(n, use_parallel_cut)` | `use_parallel_cut=True` | `src/ellalgo/ell_calc.py:48` | Enables or disables the parallel-cut optimization. |
| `LMIProblem.__init__(..., space_factory)` | `space_factory=EllStable` | `src/ellalgo/lmi_problem.py:42-46` | Selects the concrete search space (`Ell` or `EllStable`). |

Convenience constructors:

- `Ell.from_radii(val, x_center)` — `src/ellalgo/ell_base.py:101`
- `Ell.from_alpha(alpha, x_center)` — `src/ellalgo/ell_base.py:114`

Note: `EllStable.__init__(self, val, x_center)` (`src/ellalgo/ell_stable.py:48`)
overrides the base and does **not** expose `no_defer_trick`.

---

## 4. Conjugate Gradient solver

Defined in `src/ellalgo/conjugate_gradient.py:19-25`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `x0` | `None` | Initial guess; `None` means the zero vector. |
| `tol` | `1e-5` | Convergence tolerance on the residual norm. |
| `max_iter` | `1000` | Maximum iterations before raising `ConvergenceError`. |

---

## 5. Oracle parameters

These are problem/data inputs rather than tuning knobs. All are positional
arguments without code defaults unless noted.

| Oracle | Signature | Location | Defaults |
| --- | --- | --- | --- |
| `LowpassOracle` | `(ndim, wpass, wstop, lp_sq, up_sq, sp_sq)` | `src/ellalgo/oracles/lowpass_oracle.py:94` | none (all required) |
| `create_lowpass_case` | `(ndim=48)` | `src/ellalgo/oracles/lowpass_oracle.py:307` | passband `0.12`, stopband `0.20`, ripple `0.025`, attenuation `0.125` |
| `ProfitOracle` | `(params, elasticities, price_out)` | `src/ellalgo/oracles/profit_oracle.py:76` | none |
| `ProfitRbOracle` | `(params, elasticities, price_out, vparams)` | `src/ellalgo/oracles/profit_oracle.py:221` | none (`vparams` = uncertainties ε₁…ε₅) |
| `ProfitQOracle` | `(params, elasticities, price_out)` | `src/ellalgo/oracles/profit_oracle.py:287` | none |
| `LMIOracle` | `(mat_f, mat_b)` | `src/ellalgo/oracles/lmi_oracle.py:57` | none |
| `LMI0Oracle` | `(mat_f)` | `src/ellalgo/oracles/lmi0_oracle.py:46` | none |
| `LMIOldOracle` | `(mat_f, mat_b)` | `src/ellalgo/oracles/lmi_old_oracle.py:53` | none |

Factory wrappers exist in `src/ellalgo/oracles/lmi_factory.py`:
`make_lmi_oracle`, `make_lmi0_oracle`, `make_lmi_old_oracle`.

---

## 6. Utility helper

`RoundRobin.__init__(n, lo=0, hi=None, start=None)` —
`src/ellalgo/round_robin.py:36-38`.

---

## 7. Hardcoded internal constants (not exposed)

These influence numerical behavior but are not parameters; they are fixed in the
source.

| Constant | Value | Location |
| --- | --- | --- |
| `_TINY` (denormal `omega` guard) | `np.finfo(np.float64).tiny` ≈ `2.225e-308` | `src/ellalgo/ell.py:24` |
| Lowpass frequency-grid density | `mdim = 15 * ndim` | `src/ellalgo/oracles/lowpass_oracle.py:131` |
| Spectral over-sampling factor | `mult_factor = 100` | `src/ellalgo/oracles/spectral_fact.py:60` |
| Spectral log clamp | `-1e-4` triggers replacement by `1e-10` | `src/ellalgo/oracles/spectral_fact.py:81-82` |
| `EllCalcCore` derived constants (`_cst0` … `_cst3`, `_half_n`, …) | computed from `n` | `src/ellalgo/ell_calc_core.py:78-86` |

---

## 8. Values used in tests, benchmarks, and experiments

The library defaults are frequently overridden in the repository. These values
show the practical operating ranges.

| Source | `tolerance` | `max_iters` |
| --- | --- | --- |
| `experiment/power_iteration.py:149` | `1e-7` | `2000` |
| `experiment/power_iteration.py:161` | `1e-14` | `2000` |
| `benches/test_bm_lowpass.py:31-32` | `1e-14` | `50000` |
| `benches/bench_lowpass_cvxpy.py:109-110` | `1e-14` | `50000` |
| `benches/bench_chebyshev.py:66-67` | `1e-10` | `500 * (n + 1) ** 2` |
| `benches/plot_bench_chebyshev.py:57-58` | `1e-10` | `500 * (n + 1) ** 2` |
| `benches/bench_lmi_evp.py:143-144` | `1e-10` | `20000` |
| `benches/test_bm_example1.py:115` | `1e-10` | default (`2000`) |
| `demo/chebyshev_center.py:181-182` | `1e-10` | `5000` |
| `tests/test_example1.py:74` | `1e-10` | default (`2000`) |
| `tests/test_quasicvx.py:117` | `1e-8` | default (`2000`) |
| `tests/test_example3.py:87` | `1e-8` | default (`2000`) |
| `tests/integration/test_cutting_plane_workflows.py:80-88` | `1e-6` (loose) / `1e-20` (tight) | `100` / `1000` |
| `tests/test_cutting_plane.py:117-394` | various (`1e-7`, `0.0`) | `3`–`2000` |

---

## Summary

The only genuine algorithm-level tunables are:

- `Options.max_iters` (default `2000`)
- `Options.tolerance` (default `1e-20`)
- `Options.verbose` (default `False`, currently unused)
- `EllCalc.use_parallel_cut` (default `True`)
- `Ell.no_defer_trick` (default `False`)
- `conjugate_gradient`'s `tol` (default `1e-5`) and `max_iter` (default `1000`)

Everything else is problem data (oracle matrices, prices, filter
specifications) or a hardcoded numerical constant.
