"""
Ellipsoid iterations
====================

The ellipsoid search space shrinks as cutting planes are applied.
Initial ellipsoid (blue) is narrowed to the final ellipsoid (orange)
after 10 iterations of the cutting-plane method.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse as MplEllipse

from ellalgo import Ell


def ell_to_patch(ell: Ell, **kwargs):
    """Convert an Ell object to a matplotlib Ellipse patch."""
    # ellipsoid: (x - xc).T * M^(-1) * (x - xc) <= kappa^2
    # M = _mq, axes from eigenvectors, radii = kappa * sqrt(eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(ell._mq)
    # Ensure positive eigenvalues
    eigvals = np.maximum(eigvals, 1e-12)
    width = 2 * ell._kappa * np.sqrt(eigvals[0])
    height = 2 * ell._kappa * np.sqrt(eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return MplEllipse(xy=ell._xc, width=width, height=height, angle=angle, **kwargs)


# Run a simple cutting-plane feasibility problem in 2D
class SimpleOracle:
    """Feasibility: x[0] + x[1] <= 1, x[0] >= 0, x[1] >= 0"""

    def assess_feas(self, xc):
        if xc[0] + xc[1] > 1:
            return (np.array([1.0, 1.0]), 0.0)  # cut: x+y <= 1
        if xc[0] < 0:
            return (np.array([-1.0, 0.0]), 0.0)  # cut: x >= 0
        if xc[1] < 0:
            return (np.array([0.0, -1.0]), 0.0)  # cut: y >= 0
        return None  # feasible


ell = Ell(10.0, np.array([5.0, 5.0]))
snapshots = []

for _ in range(10):
    snapshots.append((ell._xc.copy(), ell._mq.copy(), ell._kappa))
    cut = SimpleOracle().assess_feas(ell._xc)
    if cut is None:
        break
    ell.update_bias_cut(cut)

fig, ax = plt.subplots(figsize=(6, 6))
# Plot initial ellipsoid
init_eigvals, _ = np.linalg.eigh(snapshots[0][1])
rad = 2 * snapshots[0][2] * np.sqrt(np.maximum(init_eigvals, 1e-12))
patch = ell_to_patch(
    type(
        "_",
        (),
        {"_xc": snapshots[0][0], "_mq": snapshots[0][1], "_kappa": snapshots[0][2]},
    )(),
    fill=False,
    edgecolor="#1565C0",
    linewidth=2,
    label="Initial",
)
ax.add_patch(patch)

# Plot final ellipsoid
final = snapshots[-1]
patch = ell_to_patch(
    type("_", (), {"_xc": final[0], "_mq": final[1], "_kappa": final[2]})(),
    fill=False,
    edgecolor="#E65100",
    linewidth=2,
    label=f"Iter {len(snapshots)}",
)
ax.add_patch(patch)

# Plot center points
init_xc = snapshots[0][0]
ax.plot(init_xc[0], init_xc[1], "o", color="#1565C0", markersize=8)
final_xc = final[0]
ax.plot(final_xc[0], final_xc[1], "s", color="#E65100", markersize=8)

# Shade feasible region
xs = np.linspace(-1, 6, 5)
ys = np.linspace(-1, 6, 5)
X, Y = np.meshgrid(xs, ys)
Z = np.maximum(0, np.maximum(X + Y - 1, np.maximum(-X, -Y)))
ax.contourf(X, Y, Z == 0, levels=[-0.5, 0.5], colors=["#C8E6C9"], alpha=0.4)

ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.set_aspect("equal")
ax.set_title("Ellipsoid shrinking via cutting planes")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.legend()
ax.grid(True, alpha=0.3)
