"""
Cutting Plane Algorithm Implementation

This code implements various cutting plane algorithms, which are optimization techniques used to solve convex optimization problems. The main purpose of these algorithms is to find optimal or feasible solutions within a given search space.

The code defines several functions that take different inputs:

1. cutting_plane_feas: Takes an oracle (a function that assesses feasibility), a search space, and options.
2. cutting_plane_optim: Takes an optimization oracle, a search space, an initial best value (gamma), and options.
3. cutting_plane_optim_q: Similar to cutting_plane_optim, but for quantized discrete optimization problems.
4. bsearch: Performs a binary search using an oracle and an interval.

These functions generally output a solution (if found), the best value achieved, and the number of iterations performed.

The algorithms work by iteratively refining the search space. They start with an initial point and ask the oracle to assess it. The oracle either confirms the point is feasible/optimal or provides a "cut" - information about how to improve the solution. The search space is then updated based on this cut, and the process repeats until a solution is found or the maximum number of iterations is reached.

An important concept in these algorithms is the "cut". A cut is like a hint that tells the algorithm which parts of the search space to exclude, helping it focus on more promising areas. The search space is continuously shrunk based on these cuts until a solution is found or the space becomes too small.

The code also includes a BSearchAdaptor class, which adapts a feasibility oracle to work with the binary search algorithm. This allows the binary search to be used in solving certain types of optimization problems.

Throughout the code, there's a focus on handling different types of problems (feasibility, optimization, discrete optimization) and different types of search spaces. The algorithms are designed to be flexible and work with various problem types.

In summary, this code provides a toolkit for solving different types of optimization problems using cutting plane methods. It's designed to be adaptable to various problem types and to efficiently search for solutions by iteratively refining the search space based on feedback from problem-specific oracles.
"""

import copy
from typing import Any, MutableSequence, Optional, Tuple, Union

from .ell_config import CutStatus, Options
from .ell_typing import (  # OracleFeasQ,
    ArrayType,
    OracleBS,
    OracleFeas,
    OracleFeas2,
    OracleOptim,
    OracleOptimQ,
    SearchSpace,
    SearchSpace2,
    SearchSpaceQ,
)

CutChoice = Union[float, MutableSequence]  # Single cut or parallel cuts
Cut = Tuple[ArrayType, CutChoice]  # Cut representation: (gradient, intercept)

Num = Union[float, int]


def cutting_plane_feas(
    omega: OracleFeas[ArrayType],
    space: SearchSpace[ArrayType],
    options=Options(),
) -> Tuple[Optional[ArrayType], int]:
    r"""Cutting-plane algorithm
    for convex feasibility problems.

    Implementation Details:
    Solves: find x s.t. f(x) ≤ 0
    for convex f(x)
    using iterative cutting planes
    - At each iteration:
      1. Query oracle at current center point xc = space.xc()
      2. If feasible (cut=None), return xc as solution
      3. If infeasible, get separating hyperplane (cut) from oracle
      4. Update search space by eliminating region violating cut
      5. Repeat until space becomes too small (tsq < tolerance)

    Mathematical Basis:
    For convex function f and current point xc:
    - If f(xc) > 0: exists g s.t. f(x) ≥ g^T(x - xc)
      + f(xc)
      > 0 for some x
    - The cut g^T(x - xc) + β ≤ 0 (β = f(xc))
      eliminates infeasible region

        .. svgbob::
           :align: center

         ┌────────────┐    ┌───────────┐┌──────────┐
         │CuttingPlane│    │SearchSpace││OracleFeas│
         └─────┬──────┘    └─────┬─────┘└────┬─────┘
               │                 │           │
               │   request xc    │           │
               │────────────────>│           │
               │                 │           │
               │    return xc    │           │
               │<────────────────│           │
               │                 │           │
               │       assess_feas(xc)       │
               │────────────────────────────>│
               │                 │           │
               │         return cut          │
               │<────────────────────────────│
               │                 │           │
               │update by the cut│           │
               │────────────────>│           │
         ┌─────┴──────┐    ┌─────┴─────┐┌────┴─────┐
         │CuttingPlane│    │SearchSpace││OracleFeas│
         └────────────┘    └───────────┘└──────────┘

    :param omega: Feasibility oracle implementing assess_feas()
    :param space: Search space object maintaining current solution candidate
    :param options: Algorithm control parameters
    :return: (Feasible solution, iteration count) or (None, iterations)

    Examples:
        >>> import numpy as np
        >>> from ellalgo.cutting_plane import cutting_plane_feas
        >>> from ellalgo.ell import Ell
        >>> from ellalgo.ell_config import Options
        >>> class MyOracle:
        ...     def assess_feas(self, xc):
        ...         return (np.array([1.0, 1.0]), 0.0) if xc[0] + xc[1] > 0 else None
        >>> omega = MyOracle()
        >>> space = Ell(10.0, np.array([0.0, 0.0]))
        >>> x, niter = cutting_plane_feas(omega, space, Options())
        >>> x is None
        False
    """
    for niter in range(options.max_iters):
        # Evaluate current solution
        cut = omega.assess_feas(space.xc())
        if cut is None:  # Found feasible point
            return space.xc(), niter
        # Update search space with new constraint
        status = space.update_bias_cut(cut)
        if status != CutStatus.Success or space.tsq() < options.tolerance:
            return None, niter
    return None, options.max_iters


def cutting_plane_optim(
    omega: OracleOptim[ArrayType],
    space: SearchSpace[ArrayType],
    gamma,
    options=Options(),
) -> Tuple[Optional[ArrayType], float, int]:
    """Cutting-plane method
    for convex optimization problems.

    Solves: maximize γ s.t. f(x) ≥ γ using central/bias cut updates
    - Maintains current best γ and candidate solution x_best
    - Alternates between:
      1. Optimality cuts (improve γ when better solution found)
      2. Feasibility cuts (maintain solution space when γ increases)

    Update Rules:
    - Central cut: Tightens search around improving solutions
    - Bias cut: Maintains feasibility for current γ level

    :param omega: Optimization oracle implementing assess_optim()
    :param space: Search space maintaining current solution candidate
    :param gamma: Initial best objective value
    :param options: Algorithm control parameters
    :return: (Best solution, achieved γ, iteration count)
    """
    x_best = None
    for niter in range(options.max_iters):
        # Get optimality/feasibility cut and possible better γ
        cut, gamma1 = omega.assess_optim(space.xc(), gamma)
        if gamma1 is not None:  # Found better objective value
            gamma = gamma1
            x_best = copy.copy(space.xc())
            status = space.update_central_cut(cut)  # Focus search around improvement
        else:
            status = space.update_bias_cut(cut)  # Maintain feasibility
        if status != CutStatus.Success or space.tsq() < options.tolerance:
            return x_best, gamma, niter
    return x_best, gamma, options.max_iters


# def cutting_plane_feas_q(
#     omega: OracleFeasQ[ArrayType], space_q: SearchSpaceQ[ArrayType], options=Options()
# ) -> Tuple[Optional[ArrayType], int]:
#     """Cutting-plane method for solving convex discrete optimization problem
#
#     :param omega: The parameter "omega" is an instance of the OracleFeasQ class, which is used to
#         perform assessments on the initial solution "xinit"
#
#     :type omega: OracleFeasQ[ArrayType]
#
#     :param space_q: The `space_q` parameter is an instance of the `SearchSpaceQ` class, which represents
#         the search space for the discrete optimization problem. It contains information about the current
#         solution candidate `x*` and provides methods for updating the search space based on the cutting
#         plane information
#
#     :type space_q: SearchSpaceQ[ArrayType]
#
#     :param options: The `options` parameter is an instance of the `Options` class, which contains
#         various options for the cutting-plane method. It is optional and has default values if not provided
#
#     :return: a tuple containing two elements:
#         1. Optional[ArrayType]: A feasible solution to the convex discrete optimization problem. If no
#         feasible solution is found, it returns None.
#         2. int: The number of iterations performed by the cutting-plane method.
#     """
#     retry = False
#     for niter in range(options.max_iters):
#         cut, x_q, more_alt = omega.assess_feas_q(space_q.xc(), retry)
#         if cut is None:  # better gamma obtained
#             return x_q, niter
#         status = space_q.update_q(cut)
#         if status == CutStatus.Success:
#             retry = False
#         elif status == CutStatus.NoSoln:
#             return None, niter
#         elif status == CutStatus.NoEffect:
#             if not more_alt:  # no more alternative cut
#                 return None, niter
#             retry = True
#         if space_q.tsq() < options.tolerance:
#             return None, niter
#     return None, options.max_iters


def cutting_plane_optim_q(
    omega: OracleOptimQ[ArrayType],
    space_q: SearchSpaceQ[ArrayType],
    gamma,
    options=Options(),
) -> Tuple[Optional[ArrayType], float, int]:
    """Cutting-plane method for discrete convex optimization.

    Handles quantized solutions through:
    - Continuous relaxation with rounding
    - Retry mechanism for discrete feasibility checks
    - Adaptive cut management for discrete solutions

    Process Flow:
    1. First attempt with continuous solution
    2. If feasible, round to nearest integer solution
    3. Verify discrete solution feasibility
    4. Generate cuts adjusted for rounding effects

    :param omega: Discrete optimization oracle
    :param space_q: Quantized search space
    :param gamma: Initial best objective value
    :param options: Algorithm parameters
    :return: (Best discrete solution, achieved γ, iterations)
    """
    x_best = None
    retry = False  # Discrete feasibility check flag
    for niter in range(options.max_iters):
        # Get cut and possible discrete solution
        cut, x_q, gamma1, more_alt = omega.assess_optim_q(space_q.xc(), gamma, retry)
        if gamma1 is not None:  # Improved objective value
            gamma = gamma1
            x_best = x_q
        # Update search space with quantized cut
        status = space_q.update_q(cut)
        if status == CutStatus.Success:
            retry = False  # Valid cut applied
        elif status == CutStatus.NoSoln:
            return x_best, gamma, niter  # No solution exists
        elif status == CutStatus.NoEffect:
            if not more_alt:  # Exhausted alternative cuts
                return x_best, gamma, niter
            retry = True  # Retry with discrete solution
        if space_q.tsq() < options.tolerance:
            return x_best, gamma, niter
    return x_best, gamma, options.max_iters


def bsearch(
    omega: OracleBS, intrvl: Tuple[Any, Any], options=Options()
) -> Tuple[Any, int]:
    """Binary search with feasibility oracle.

    Operates on monotonic objectives by:
    1. Maintaining upper/lower bounds
    2. Testing mid-point feasibility
    3. Halving search interval each iteration

    :param omega: Binary search oracle implementing assess_bs()
    :param intrvl: (lower, upper) bound tuple
    :param options: Control parameters
    :return: (Best γ found, iterations used)
    """
    lower, upper = intrvl
    T = type(upper)  # Preserve numerical type (int/float)
    for niter in range(options.max_iters):
        tau = (upper - lower) / 2
        if tau < options.tolerance:  # Convergence check
            return upper, niter
        gamma = T(lower + tau)
        if omega.assess_bs(gamma):  # Feasible -> move upper bound down
            upper = gamma
        else:  # Infeasible -> move lower bound up
            lower = gamma
    return upper, options.max_iters


class BSearchAdaptor(OracleBS):
    """Adapter for using feasibility oracle in binary search.

    Enables γ-parameterized feasibility checks for:
    maximize γ
    s.t. ∃x feasible for given γ

    Maintains state between binary search iterations for efficiency.
    """

    def __init__(
        self, omega: OracleFeas2, space: SearchSpace2, options=Options()
    ) -> None:
        """
        :param omega: γ-parameterized feasibility oracle
        :param space: Search space for feasibility subproblems
        :param options: Cutting-plane parameters for subproblems
        """
        self.omega = omega  # Gamma-sensitive feasibility oracle
        self.space = space  # Search space for subproblems
        self.options = options  # Subproblem solver parameters

    @property
    def x_best(self):
        """Current best feasible solution candidate."""
        return self.space.xc()

    def assess_bs(self, gamma: Num) -> bool:
        """Test feasibility for given γ value.

        Implementation:
        1. Clone current search space state
        2. Update oracle with new γ value
        3. Solve feasibility subproblem
        4. Update main space if feasible solution found
        """
        space = copy.deepcopy(self.space)  # Isolate subproblem space
        self.omega.update(gamma)  # Set current γ level
        x_feas, _ = cutting_plane_feas(self.omega, space, self.options)
        if x_feas is not None:  # Feasible solution found
            self.space.set_xc(x_feas)  # Update main search space
            return True
        return False
