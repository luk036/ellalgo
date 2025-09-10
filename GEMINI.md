# Project Overview

This is a Python project that implements the Ellipsoid Method for convex optimization. The project is named `ellalgo` and is set up using PyScaffold. The main logic is contained in the `src/ellalgo` directory.

The Ellipsoid Method is an iterative algorithm for solving linear programming and convex optimization problems. It works by enclosing the feasible region within an ellipsoid and successively shrinking the ellipsoid until it converges to the optimal solution.

## Building and Running

### Dependencies

The project's dependencies are listed in `setup.cfg`. The main dependency is `numpy`.

### Building the Package

To build the package, you can use the following command:

```bash
tox -e build
```

This will create a `dist` directory with the built package.

### Running Tests

The project uses `pytest` for testing. To run the tests, use the following command:

```bash
tox
```

This will run the tests defined in the `tests` directory.

## Development Conventions

### Code Style

The project uses `black` for code formatting and `isort` for sorting imports. `flake8` is used for linting. These are enforced using pre-commit hooks, which are configured in `.pre-commit-config.yaml`.

### Committing Code

Before committing any code, make sure to run the pre-commit hooks to ensure that the code is properly formatted and linted.

```bash
pre-commit run --all-files
```

### Documentation

The project's documentation is built using Sphinx and is located in the `docs` directory. To build the documentation, you can use the following command:

```bash
tox -e docs
```
