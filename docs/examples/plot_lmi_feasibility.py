"""
LMI feasibility
===============

Solving a 2D Linear Matrix Inequality using the ellipsoid method.
The ellipsoid (blue) shrinks toward the feasible region (green).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse as MplEllipse

from ellalgo import Ell, cutting_plane_feas
from ellalgo.oracles.lmi_oracle import LMIOracle

# LMI constraint: B - (F1*x1 + F2*x2) >= 0
# B = [[2, 0], [0, 1]], F1 = [[1, 0], [0, 0]], F2 = [[0, 0], [0, 1]]
B = np.array([[2.0, 0.0], [0.0, 1.0]])
F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
F2 = np.array([[0.0, 0.0], [0.0, 1.0]])

omega = LMIOracle([F1, F2], B)
space = Ell(5.0, np.array([0.0, 0.0]))

# Take snapshots during iterations
snapshots = []
ell = space  # reference
for _ in range(15):
    snapshots.append((ell._xc.copy(), ell._mq.copy(), ell._kappa))
    cut = omega.assess_feas(ell._xc)
    if cut is None:
        break
    ell.update_bias_cut(cut)

x_best, _ = cutting_plane_feas(omega, space)

fig, ax = plt.subplots(figsize=(6, 6))

# Plot selected snapshots (fading colors)
n = len(snapshots)
for i, (xc, mq, kappa) in enumerate(snapshots):
    alpha = 0.3 + 0.7 * (i / n) if n > 1 else 0.6
    eigvals, eigvecs = np.linalg.eigh(mq)
    eigvals = np.maximum(eigvals, 1e-12)
    w = 2 * kappa * np.sqrt(eigvals[0])
    h = 2 * kappa * np.sqrt(eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    patch = MplEllipse(
        xy=xc,
        width=w,
        height=h,
        angle=angle,
        fill=False,
        edgecolor="#1565C0",
        linewidth=1.2,
        alpha=alpha,
    )
    ax.add_patch(patch)

# Plot center trajectory
centers = np.array([s[0] for s in snapshots])
ax.plot(
    centers[:, 0],
    centers[:, 1],
    "o-",
    color="#1565C0",
    markersize=4,
    linewidth=0.8,
    alpha=0.6,
    label="Ellipsoid centers",
)

# Mark final solution
if x_best is not None:
    ax.plot(
        x_best[0],
        x_best[1],
        "*",
        color="#E65100",
        markersize=15,
        label=f"Solution ({x_best[0]:.2f}, {x_best[1]:.2f})",
    )

ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect("equal")
ax.set_title("LMI Feasibility via Ellipsoid Method (15 iterations)")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.legend()
ax.grid(True, alpha=0.3)
