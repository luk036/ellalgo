"""
Factory functions for the LMI oracle family.

Provides uniform construction entry points for the three LMI oracle variants:
LMIOracle (lazy), LMI0Oracle (no constant term), and LMIOldOracle (explicit).
"""

from typing import List

import numpy as np

from ellalgo.oracles.lmi0_oracle import LMI0Oracle
from ellalgo.oracles.lmi_old_oracle import LMIOldOracle
from ellalgo.oracles.lmi_oracle import LMIOracle


def make_lmi_oracle(mat_f: List[np.ndarray], mat_b: np.ndarray) -> LMIOracle:
    """Create an LMIOracle (lazy matrix form).

    Args:
        mat_f: List of symmetric coefficient matrices [F₁, F₂, ..., Fₙ].
        mat_b: Constant matrix B defining the LMI constraint B − ΣFₖxₖ ⪰ 0.

    Returns:
        A configured LMIOracle.
    """
    return LMIOracle(mat_f, mat_b)


def make_lmi0_oracle(mat_f: List[np.ndarray]) -> LMI0Oracle:
    """Create an LMI0Oracle (compact form, no constant term).

    Args:
        mat_f: List of symmetric coefficient matrices [F₁, F₂, ..., Fₙ].

    Returns:
        A configured LMI0Oracle.
    """
    return LMI0Oracle(mat_f)


def make_lmi_old_oracle(mat_f: List[np.ndarray], mat_b: np.ndarray) -> LMIOldOracle:
    """Create an LMIOldOracle (explicit matrix form).

    Args:
        mat_f: List of symmetric coefficient matrices [F₁, F₂, ..., Fₙ].
        mat_b: Constant matrix B defining the LMI constraint B − ΣFₖxₖ ⪰ 0.

    Returns:
        A configured LMIOldOracle.
    """
    return LMIOldOracle(mat_f, mat_b)
