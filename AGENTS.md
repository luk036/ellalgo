# AGENTS.md

This document provides guidelines for agentic coding assistants working on the `ellalgo` repository.

## Build, Lint, and Test Commands

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ell.py

# Run specific test function
pytest tests/test_ell.py::test_construct

# Run with coverage
pytest --cov=ellalgo --cov-report term-missing

# Run doctests
pytest --doctest-modules src/ellalgo/

# Run benchmarks (requires pytest-benchmark)
pytest benches/
```

### Linting and Formatting

```bash
# Run pre-commit hooks (includes isort, black, flake8)
pre-commit run --all-files

# Individual tools:
isort .               # Sort imports
black .               # Format code
flake8 .              # Lint code (max_line_length=256, ignores E203, W503)

# Fix linting issues automatically where possible
black . && isort .
```

### Building the Package

```bash
# Build wheel and sdist
python -m build

# Clean build artifacts
python -c 'import shutil; [shutil.rmtree(p, True) for p in ("build", "dist", "docs/_build")]'
```

### Documentation

```bash
# Build docs
sphinx-build -b html docs/ docs/_build/html

# Run doctests in docs
sphinx-build -b doctest docs/ docs/_build/doctest
```

## Code Style Guidelines

### Imports

**Import Order**: Standard library → Third-party → Local (ellalgo)

```python
# 1. Standard library
import copy
import math
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

# 2. Third-party
import numpy as np

# 3. Local imports (prefer relative for same package)
from .ell_calc import EllCalc
from .ell_config import CutStatus, Options
from ellalgo.oracles.ldlt_mgr import LDLTMgr  # Absolute for other packages
```

**Patterns**:
- Use `from __future__ import annotations` for forward references
- Group imports by blank lines
- Avoid wildcard imports (`from module import *`)

### Type Hints

**Comprehensive type annotations required**:

```python
from typing import Generic, List, Optional, Tuple, TypeVar, Union

ArrayType = TypeVar("ArrayType", bound=np.ndarray)

def function_name(
    param1: ArrayType,
    param2: Union[float, int],
    options: Options = Options(),
) -> Tuple[Optional[ArrayType], int]:
    pass

class MyClass(Generic[ArrayType]):
    _private_attr: np.ndarray
    public_attr: float = 1.0
```

**Type aliases** (defined at module level):
```python
ArrayType = TypeVar("ArrayType", bound=np.ndarray)
CutChoice = Union[float, List[float], Tuple[float, ...]]
Cut = Tuple[ArrayType, CutChoice]
Num = Union[float, int]
Mat = np.ndarray
```

### Naming Conventions

**Classes**: PascalCase
```python
class EllipsoidCalculator:
    pass

class CutStatus(Enum):
    Success = 0
    NoSoln = 1
```

**Functions and Methods**: snake_case
```python
def cutting_plane_feas(omega, space, options):
    pass

def update_central_cut(self, cut):
    pass
```

**Private Members**: _snake_case
```python
class Ell:
    _xc: ArrayType  # Private instance variable
    _mq: Mat

    def _update_core(self, cut):  # Private method
        pass
```

**Parameters and Variables**: snake_case
```python
x_center, cut, gamma, tsq, kappa
```

**Type Aliases**: PascalCase (most common)
```python
ArrayType, CutChoice, Cut, Mat
```

### Error Handling

**No try/except in library code** - use explicit validation:

```python
# ❌ DON'T
try:
    result = risky_operation()
except:
    pass

# ✅ DO
if invalid_condition:
    raise ValueError("Descriptive error message")
```

**Use status enums for algorithmic outcomes**:
```python
# Return CutStatus instead of raising exceptions
status = ell.update_central_cut(cut)
if status == CutStatus.NoSoln:
    # Handle infeasible case
```

**Raise ValueError for invalid inputs**:
```python
if np.all(grad == 0.0):
    raise ValueError("Gradient cannot be a zero vector.")
```

### Documentation

**Google-style docstrings**:

```python
def function_name(param: type) -> return_type:
    """Brief one-line description.

    Optional detailed multi-line description explaining the function's purpose,
    mathematical basis, or implementation details.

    Args:
        param (type): Description of parameter

    Returns:
        type: Description of return value

    Raises:
        ValueError: When and why this exception occurs

    Examples:
        >>> from ellalgo.ell import Ell
        >>> ell = Ell(1.0, np.array([0.0, 0.0]))
        >>> ell.xc()
        array([0., 0.])
    """
```

**Requirements**:
- First line: Brief summary
- **Args**: Parameter descriptions with types
- **Returns**: Return value with type
- **Raises**: Exception conditions (only when applicable)
- **Examples**: Doctest-style examples using `>>>`
- Use `r"""` for raw strings when including LaTeX notation

**Module-level docstrings**: Explain purpose and theory

```python
"""
Cutting Plane Algorithm Implementation

This code implements various cutting plane algorithms...

[Detailed explanation of algorithm theory and approach]
"""
```

### Code Organization

**Package Structure**:
```
src/ellalgo/
├── __init__.py           # Package initialization, version info
├── ell.py                # Main Ellipsoid class
├── ell_calc.py           # Ellipsoid calculator
├── ell_config.py         # CutStatus enum, Options class
├── ell_typing.py         # Type aliases and abstract base classes
├── cutting_plane.py      # Cutting plane algorithms
├── oracles/              # Oracle implementations
│   ├── lmi_oracle.py
│   ├── profit_oracle.py
│   └── ...
└── tests/                # Unit tests
```

**Abstract Base Classes**: Defined in `ell_typing.py`
```python
class OracleFeas(Generic[ArrayType]):
    @abstractmethod
    def assess_feas(self, x_center: ArrayType) -> Optional[Cut]:
        pass
```

### Testing Patterns

**Use pytest** with property-based testing (hypothesis) where applicable:

```python
import numpy as np
import pytest
from pytest import approx
from hypothesis import given, strategies as st

def test_construct() -> None:
    ell = Ell(0.01, np.zeros(4))
    assert ell._kappa == 0.01
    assert ell._xc == approx(np.zeros(4))

@given(st.floats(min_value=0.01, max_value=10.0))
def test_positive_kappa(kappa: float) -> None:
    ell = Ell(kappa, np.zeros(4))
    assert ell._kappa == approx(kappa)
```

**Test naming**: `test_<function_or_method>_<scenario>`

### Performance

- Use `numpy` for all numerical computations
- Prefer vectorized operations over loops
- Benchmark with `pytest-benchmark` in `benches/` directory

### Profiling

For performance analysis and optimization:

```bash
# Profile a specific test
python -m cProfile -o profile.stats -m pytest tests/test_large_problem.py

# Analyze profiling results
python -m pstats profile.stats

# Profile with snakeviz for visualization
python -m cProfile -o profile.prof -m pytest tests/test_large_problem.py
pip install snakeviz
snakeviz profile.prof
```

### Configuration

**Formatter**: Black (23.7.0)
- Line length: 256 (configured in setup.cfg)
- Black-compatible flake8 settings (ignores E203, W503)

**Import Sorter**: isort (5.12.0)

**Linter**: flake8 (6.1.0)
- Max line length: 256
- Extends ignore: E203, W503

**Pre-commit**: Configured in `.pre-commit-config.yaml`

## Key Concepts

- **Cutting Plane Methods**: Iterative optimization algorithms using cuts to narrow search space
- **Ellipsoid**: Geometric search space represented by center and shape matrix
- **Oracle**: Function that assesses feasibility/optimality and returns cuts
- **Cut**: A hyperplane that separates feasible from infeasible regions
- **CutStatus**: Enum indicating outcome of a cut operation

When working on this codebase, follow the established patterns, maintain type safety, and document all public APIs with comprehensive docstrings.
