"""
Plot run-time comparisons for the two new experiments:
  1. LMI min-eigenvalue EVP (CVXPY vs Python ellipsoid, m = 5..20)
  2. FIR lowpass design (CVXPY vs Python ellipsoid vs C++ ellipsoid, n = 24..80)

Uses measured numbers from this session's benchmarks:
  - LMI: benches/bench_lmi_evp.py
  - Lowpass: benches/bench_lowpass_cvxpy.py + ellalgo-cpp BM_lowpass_cvxpy.exe

Saves benches/bench_lmi_evp.svg and benches/bench_lowpass3.svg.
"""

import matplotlib.pyplot as plt

# --- LMI min-eigenvalue EVP (m = 5..20, vars = m+1) ---
lmi_m = [5, 8, 12, 16, 20]
lmi_cvxpy = [0.0316, 0.0182, 0.0201, 0.0208, 0.0258]
lmi_py = [0.1122, 0.2664, 0.5480, 1.1686, 0.5061]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(lmi_m, lmi_cvxpy, "o-", label="CVXPY (CLARABEL)", color="#1565C0")
ax.plot(lmi_m, lmi_py, "s-", label="Ellipsoid method (Python)", color="#E65100")
ax.set_yscale("log")
ax.set_xlabel("design variables m + 1")
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title("Min-eigenvalue LMI (EVP): run-time comparison")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("benches/bench_lmi_evp.svg", format="svg")
print("saved benches/bench_lmi_evp.svg")

# --- FIR lowpass design (n = 24..80) ---
lp_n = [24, 32, 48, 64, 80]
lp_cvxpy = [0.092, 0.135, 0.446, 0.910, 3.473]
lp_py = [2.567, 4.402, 3.433, 2.629, 4.277]
lp_cpp = [0.049037, 0.108709, 0.117180, 0.097270, 0.191468]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(lp_n, lp_cvxpy, "o-", label="CVXPY (CLARABEL)", color="#1565C0")
ax.plot(lp_n, lp_py, "s-", label="Ellipsoid method (Python)", color="#E65100")
ax.plot(lp_n, lp_cpp, "^-", label="Ellipsoid method (C++)", color="#2E7D32")
ax.set_yscale("log")
ax.set_xlabel("filter length n  (autocorrelation coefficients)")
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title("FIR lowpass design: run-time comparison (n = 24..80)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("benches/bench_lowpass3.svg", format="svg")
print("saved benches/bench_lowpass3.svg")
