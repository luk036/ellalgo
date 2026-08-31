"""
Plot the 3-way run-time comparison for the min-eigenvalue LMI (EVP) experiment
after the C++ port: CVXPY vs Python ellipsoid vs C++ ellipsoid (m = 5..20).

Uses measured numbers from this session's benchmarks:
  - CVXPY / Python ellipsoid: benches/bench_lmi_evp.py
  - C++ ellipsoid: ellalgo-cpp BM_lmi_evp.exe (nanobench median)

Saves benches/bench_lmi_evp3.svg.
"""

import matplotlib.pyplot as plt

ms = [5, 8, 12, 16, 20]
cvxpy = [0.0316, 0.0182, 0.0201, 0.0208, 0.0258]
py_ell = [0.1122, 0.2664, 0.5480, 1.1686, 0.5061]
cpp_ell = [0.000587, 0.001433, 0.004078, 0.008815, 0.029870]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(ms, cvxpy, "o-", label="CVXPY (CLARABEL)", color="#1565C0")
ax.plot(ms, py_ell, "s-", label="Ellipsoid method (Python)", color="#E65100")
ax.plot(ms, cpp_ell, "^-", label="Ellipsoid method (C++)", color="#2E7D32")
ax.set_yscale("log")
ax.set_xlabel("design variables m + 1")
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title("Min-eigenvalue LMI (EVP): run-time comparison (m = 5..20)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("benches/bench_lmi_evp3.svg", format="svg")
print("saved benches/bench_lmi_evp3.svg")
