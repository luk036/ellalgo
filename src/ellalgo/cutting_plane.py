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
from .ell_typing import (
    ArrayType,
    OracleBS,
    OracleFeas,
    OracleFeas2,
    # OracleFeasQ,
    OracleOptim,
    OracleOptimQ,
    SearchSpace,
    SearchSpace2,
    SearchSpaceQ,
)

CutChoice = Union[float, MutableSequence]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]

Num = Union[float, int]


def cutting_plane_feas(
    omega: OracleFeas[ArrayType], space: SearchSpace[ArrayType], options=Options()
) -> Tuple[Optional[ArrayType], int]:
    r"""Find a point in a convex set (defined through a cutting-plane oracle).

    Description:
        A function f(x) is *convex* if there always exist a g(x)
        such that f(z) >= f(x) + g(x)^T * (z - x), forall z, x in dom f.
        Note that dom f does not need to be a convex set in our definition.
        The affine function g^T (x - xc) + beta is called a cutting-plane,
        or a "cut" for short.
        This algorithm solves the following feasibility problem:

                find x
                s.t. f(x) <= 0,

        A *separation oracle* asserts that an evalution point x0 is feasible,
        or provide a cut that separates the feasible region and x0.

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

    :param omega: The parameter `omega` is an instance of the `OracleFeas` class, which is responsible
        for performing assessments on a given point `xinit`. It provides information about the feasibility
        of `xinit` and returns a cutting-plane (or a "cut") if `xinit` is not

    :type omega: OracleFeas[ArrayType]

    :param space: The `space` parameter represents the search space in which the algorithm is looking
        for a feasible solution. It is an instance of the `SearchSpace` class, which contains information
        about the current state of the search space, such as the current evaluation point `xc()` and the
        current trust region radius.

    :type space: SearchSpace[ArrayType]

    :param options: The `options` parameter is an instance of the `Options` class, which contains
        various options for the cutting-plane feasibility algorithm. It is optional and has default values
        if not provided

    :return: The function `cutting_plane_feas` returns a tuple containing two elements:
        1. An optional array (`Optional[ArrayType]`) representing a feasible solution. If no feasible
        solution is found, it returns `None`.
        2. An integer representing the number of iterations performed.
    """

    for niter in range(options.max_iters):
        cut = omega.assess_feas(space.xc())  # query the oracle at space.xc()
        if cut is None:  # feasible solution obtained
            return space.xc(), niter
        status = space.update_bias_cut(cut)  # update space
        if status != CutStatus.Success or space.tsq() < options.tolerance:
            return None, niter
    return None, options.max_iters


def cutting_plane_optim(
    omega: OracleOptim[ArrayType],
    space: SearchSpace[ArrayType],
    gamma,
    options=Options(),
) -> Tuple[Optional[ArrayType], float, int]:
    """Cutting-plane method for solving convex optimization problem

    :param omega: The `omega` parameter is an instance of the `OracleOptim` class, which is responsible
        for performing assessments on the initial solution `xinit`

    :type omega: OracleOptim[ArrayType]

    :param space: The `space` parameter represents the search space for the optimization problem. It is
        an instance of the `SearchSpace` class, which contains information about the feasible region and the
        current solution

    :type space: SearchSpace[ArrayType]

    :param gamma: The parameter `gamma` represents the initial best-so-far value in the cutting-plane
        optimization algorithm. It is used to keep track of the current best objective value found during
        the optimization process

    :param options: The `options` parameter is an instance of the `Options` class, which contains
        various options for the cutting-plane optimization algorithm. It is optional and has default values
        if not provided

    :return: The function `cutting_plane_optim` returns a tuple containing the following elements:
    """
    x_best = None
    for niter in range(options.max_iters):
        cut, gamma1 = omega.assess_optim(space.xc(), gamma)
        if gamma1 is not None:  # better gamma obtained
            gamma = gamma1
            x_best = copy.copy(space.xc())
            status = space.update_central_cut(cut)
        else:
            status = space.update_bias_cut(cut)
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
    """Cutting-plane method for solving convex quantized discrete optimization problem

    :param omega: The `omega` parameter is an instance of the `OracleOptimQ` class, which is responsible
        for performing assessments on the initial solution `xinit`

    :type omega: OracleOptimQ[ArrayType]

    :param space_q: The `space_q` parameter represents the search space for the discrete optimization
        problem. It is an instance of the `SearchSpaceQ` class, which contains the necessary methods and
        attributes to define and manipulate the search space

    :type space_q: SearchSpaceQ[ArrayType]

    :param gamma: The parameter `gamma` represents the initial best-so-far value in the cutting-plane
        optimization algorithm. It is used to keep track of the current best solution found during the
        iterations of the algorithm

    :param options: The `options` parameter is an instance of the `Options` class, which contains
        various settings for the cutting-plane optimization algorithm. It is optional and has default values
        if not provided

    :return: a tuple containing the following elements:
        1. Optional[ArrayType]: The optimal solution to the convex discrete optimization problem. It can be
        None if no solution is found.
        2. float: The final best-so-far value.
        3. int: The number of iterations performed.
    """
    x_best = None
    retry = False
    for niter in range(options.max_iters):
        cut, x_q, gamma1, more_alt = omega.assess_optim_q(space_q.xc(), gamma, retry)
        if gamma1 is not None:  # better gamma obtained
            gamma = gamma1
            x_best = x_q
        status = space_q.update_q(cut)
        if status == CutStatus.Success:
            retry = False
        elif status == CutStatus.NoSoln:
            return x_best, gamma, niter
        elif status == CutStatus.NoEffect:
            if not more_alt:  # no more alternative cut
                return x_best, gamma, niter
            retry = True
        if space_q.tsq() < options.tolerance:
            return x_best, gamma, niter
    return x_best, gamma, options.max_iters


def bsearch(
    omega: OracleBS, intrvl: Tuple[Any, Any], options=Options()
) -> Tuple[Any, int]:
    """
    The `bsearch` function performs a binary search to find the optimal solution within a given interval.

    :param omega: The parameter `omega` is an instance of the `OracleBS` class. It represents an oracle
        that provides information about the feasibility of a given solution. The `OracleBS` class likely has
        a method called `assess_bs` that takes a solution as input and returns a boolean value.

    :type omega: OracleBS

    :param intrvl: The `intrvl` parameter is a tuple representing the interval within which the binary
        search will be performed. It consists of two elements: the lower bound and the upper bound of the
        interval.

    :type intrvl: Tuple[Any, Any]

    :param options: The `options` parameter is an instance of the `Options` class. It is optional and
        has default values if not provided.

    :return: The function `bsearch` returns a tuple containing three elements:
    """
    # Assume monotonicity of the objective function
    lower, upper = intrvl
    T = type(upper)  # T could be `int`
    for niter in range(options.max_iters):
        tau = (upper - lower) / 2
        if tau < options.tolerance:
            return upper, niter
        gamma = T(lower + tau)
        if omega.assess_bs(gamma):  # feasible solution obtained
            upper = gamma
        else:
            lower = gamma
    return upper, options.max_iters


class BSearchAdaptor(OracleBS):
    def __init__(
        self, omega: OracleFeas2, space: SearchSpace2, options=Options()
    ) -> None:
        """
        This function initializes an object with an OracleFeas2 instance, a SearchSpace2 instance, and
        an Options instance.

        :param omega: An instance of the OracleFeas2 class. It is used to perform feasibility checks on candidate solutions

        :type omega: OracleFeas2

        :param space: The `space` parameter is an instance of the `SearchSpace2` class. It represents
            the search space in which the optimization algorithm will search for the optimal solution

        :type space: SearchSpace2

        :param options: An instance of the Options class, which contains various options and settings for the algorithm
        """
        self.omega = omega
        self.space = space
        self.options = options

    @property
    def x_best(self) -> ArrayType:
        """
        The `x_best` property returns the current best solution in the `space` object.
        :return: The `x_best` property returns an object of type `ArrayType`.
        """
        return self.space.xc()

    def assess_bs(self, gamma: Num) -> bool:
        """
        The function assess_bs checks if a given value is the best-so-far optimal value.

        :param gamma: Gamma is a float value representing the best-so-far optimal value
        :type gamma: Num
        :return: The function assess_bs returns a boolean value.
        """
        space = copy.deepcopy(self.space)
        self.omega.update(gamma)
        x_feas, _ = cutting_plane_feas(self.omega, space, self.options)
        if x_feas is not None:
            self.space.set_xc(x_feas)
            return True
        return False
