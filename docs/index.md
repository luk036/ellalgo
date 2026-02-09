# 🫒 ellalgo

> Ellipsoid Method in Python

## Quick Start

### Installation

```bash
pip install ellalgo
```

Or install from source:

```bash
git clone https://github.com/luk036/ellalgo.git
cd ellalgo
pip install -e .
```

### Basic Usage

```python
import numpy as np
from ellalgo import Ell, cutting_plane_feas
from ellalgo.oracles.lmi_oracle import LMIOracle

# Define LMI constraint: B - F1*x1 - F2*x2 >= 0
B = np.array([[2.0, 0.0], [0.0, 1.0]])
F1 = np.array([[1.0, 0.0], [0.0, 0.0]])
F2 = np.array([[0.0, 0.0], [0.0, 1.0]])

# Create oracle
omega = LMIOracle([F1, F2], B)

# Initialize ellipsoid search space
space = Ell(1.0, np.array([0.0, 0.0]))

# Run cutting plane algorithm
x_best, iter_info = cutting_plane_feas(omega, space)

print(f"Solution: {x_best}")
print(f"Iterations: {iter_info}")
```

### Core Concepts

**Ellipsoid**: A geometric search space represented by a center and shape matrix.

**Oracle**: A function that assesses feasibility/optimality and returns cuts (hyperplanes that separate feasible from infeasible regions).

**Cutting Plane Method**: An iterative optimization algorithm that uses cuts to progressively narrow the search space.

**Parallel Cut**: A pair of parallel hyperplanes used simultaneously to improve convergence.

## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

## Learn More

- [Presentation Slides](https://luk036.github.io/cvx)

[MyST]: https://myst-parser.readthedocs.io/en/latest/
