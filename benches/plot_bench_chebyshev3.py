"""
Plot 3-way run-time comparison: C++ ellipsoid vs Python ellipsoid vs CVXPY
(n = 5, 10, 15, 20, 30).

Uses the measured numbers from this session's benchmarks:
  - CVXPY (CLARABEL): benches/bench_chebyshev.py
  - Python ellipsoid: benches/bench_chebyshev.py
  - C++ ellipsoid: BM_chebyshev.exe (nanobench median)

Saves the figure to ``benches/bench_chebyshev3.svg``.
"""

import matplotlib.pyplot as plt

ns = [5, 10, 15, 20, 30]
cvxpy = [0.0580, 0.0785, 0.1365, 0.1571, 0.2543]
py_ell = [0.0619, 0.2305, 0.5001, 1.2116, 2.3567]
cpp_ell = [0.000548, 0.002348, 0.007284, 0.015754, 0.060679]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(ns, cvxpy, "o-", label="CVXPY (CLARABEL)", color="#1565C0")
ax.plot(ns, py_ell, "s-", label="Ellipsoid method (Python)", color="#E65100")
ax.plot(ns, cpp_ell, "^-", label="Ellipsoid method (C++)", color="#2E7D32")
ax.set_yscale("log")
ax.set_xlabel("dimension n  (design variables = n + 1)")
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title("Run-time comparison: Chebyshev center problem (n = 5..30)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("benches/bench_chebyshev3.svg", format="svg")
print("Figure saved to benches/bench_chebyshev3.svg")
