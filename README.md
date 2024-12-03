<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/ellalgo.svg?branch=main)](https://cirrus-ci.com/github/<USER>/ellalgo)
[![ReadTheDocs](https://readthedocs.org/projects/ellalgo/badge/?version=latest)](https://ellalgo.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/ellalgo/main.svg)](https://coveralls.io/r/<USER>/ellalgo)
[![PyPI-Server](https://img.shields.io/pypi/v/ellalgo.svg)](https://pypi.org/project/ellalgo/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/ellalgo.svg)](https://anaconda.org/conda-forge/ellalgo)
[![Monthly Downloads](https://pepy.tech/badge/ellalgo/month)](https://pepy.tech/project/ellalgo)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/ellalgo)
[![Coverage Status](https://coveralls.io/repos/github/luk036/ellalgo/badge.svg?branch=main)](https://coveralls.io/github/luk036/ellalgo?branch=main)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![Documentation Status](https://readthedocs.org/projects/ellalgo/badge/?version=latest)](https://ellalgo.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/luk036/ellalgo/branch/main/graph/badge.svg?token=6lpjUzPavX)](https://codecov.io/gh/luk036/ellalgo)

<p align="center">
  <img src="./ellipsoid-method-for-convex-optimization.svg"/>
</p>

# 🫒 ellalgo

> Ellipsoid Method in Python

The Ellipsoid Method as a linear programming algorithm was first introduced by L. G. Khachiyan in 1979. It is a polynomial-time algorithm that uses ellipsoids to iteratively reduce the feasible region of a linear program until an optimal solution is found. The method works by starting with an initial ellipsoid that contains the feasible region, and then successively shrinking the ellipsoid until it contains the optimal solution. The algorithm is guaranteed to converge to an optimal solution in a finite number of steps.

The method has a wide range of practical applications in operations research. It can be used to solve linear programming problems, as well as more general convex optimization problems. The method has been applied to a variety of fields, including economics, engineering, and computer science. Some specific applications of the Ellipsoid Method include portfolio optimization, network flow problems, and the design of control systems. The method has also been used to solve problems in combinatorial optimization, such as the traveling salesman problem.

## What is Parallel Cut?

In the context of the Ellipsoid Method, a parallel cut refers to a pair of linear constraints of the form aTx <= b and -aTx <= -b, where a is a vector of coefficients and b is a scalar constant. These constraints are said to be parallel because they have the same normal vector a, but opposite signs. When a parallel cut is encountered during the Ellipsoid Method, both constraints can be used simultaneously to generate a new ellipsoid. This can improve the convergence rate of the method, especially for problems with many parallel constraints.

## Used by

- [netoptim](https://github.com/luk036/netoptim)
- [corr-solver](https://github.com/luk036/corr-solver)
- [multiplierless](https://github.com/luk036/multiplierless)

<!-- pyscaffold-notes -->

## See also

- [Presentation Slides](https://luk036.github.io/cvx)

## 👉 Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
