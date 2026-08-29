"""
LMI problem facade: owns data and drives the cutting-plane loop.

The `LMIProblem` class bundles the LMI coefficient matrices with a lazily
created oracle and drives the cutting-plane feasibility method, hiding the
3-step recipe (build oracle -> build search space -> call driver) behind a
single call.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import cutting_plane_feas
from ellalgo.ell_config import Options
from ellalgo.ell_stable import EllStable
from ellalgo.oracles.lmi_oracle import LMIOracle


class LMIProblem:
    """LMI feasibility problem facade.

    Owns the LMI data (``mat_f`` and the constant term ``mat_b``) and the
    lazily-created LMIOracle, then drives the cutting-plane method through
    the standard ``cutting_plane_feas`` driver.

    The LMI feasibility problem is:

        find  x
        s.t.  B − Σₖ Fₖ xₖ ⪰ 0   (positive semidefinite)

    Args:
        mat_f: List of symmetric coefficient matrices [F₁, F₂, ..., Fₙ].
        mat_b: Constant matrix B defining the LMI constraint.
    """

    def __init__(self, mat_f: List[np.ndarray], mat_b: np.ndarray) -> None:
        self.mat_f = mat_f
        self.mat_b = mat_b
        self.omega = LMIOracle(mat_f, mat_b)

    def solve_feas(
        self,
        val: float | np.ndarray,
        x_center: np.ndarray,
        options: Options = Options(),
    ) -> Tuple[Optional[np.ndarray], int]:
        """Solve the LMI feasibility problem.

        Builds an EllStable search space with the given initial ellipsoid
        parameters and runs the cutting-plane feasibility method.

        Args:
            val: Either a scalar (kappa) or per-axis values for the initial
                ellipsoid.
            x_center: Initial center point.
            options: Algorithm control parameters.

        Returns:
            Tuple (solution x, number of iterations). The solution is None
            if no feasible point was found.

        Examples:
            >>> import numpy as np
            >>> F1 = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> F2 = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> B = np.array([[2.0, 0.0], [0.0, 2.0]])
            >>> problem = LMIProblem([F1, F2], B)
            >>> x, niter = problem.solve_feas(10.0, np.zeros(2))
            >>> x is not None
            True
        """
        space = EllStable(val, x_center)
        return cutting_plane_feas(self.omega, space, options)
