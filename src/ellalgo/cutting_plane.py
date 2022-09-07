# -*- coding: utf-8 -*-
from enum import Enum
from typing import Any, Tuple


class CutStatus(Enum):
    Success = 0
    NoSoln = 1
    SmallEnough = 2
    NoEffect = 3
    Unknown = 4


class Options:
    max_iter: int = 2000  # maximum number of iterations
    tol: float = 1e-8  # error tolerance


class CInfo:
    def __init__(self, feasible: bool, num_iters: int, status: CutStatus):
        """Construct a new CInfo object

        Arguments:
            feasible (bool): [description]
            num_iters (int): [description]
            status (int): [description]
        """
        self.feasible: bool = feasible
        self.num_iters: int = num_iters
        self.status: CutStatus = status


def cutting_plane_feas(omega, S, options=Options()) -> CInfo:
    """Find a point in a convex set (defined through a cutting-plane oracle).

    Description:
        A function f(x) is *convex* if there always exist a g(x)
        such that f(z) >= f(x) + g(x)' * (z - x), forall z, x in dom f.
        Note that dom f does not need to be a convex set in our definition.
        The affine function g' (x - xc) + beta is called a cutting-plane,
        or a ``cut'' for short.
        This algorithm solves the following feasibility problem:

                find x
                s.t. f(x) <= 0,

        A *separation oracle* asserts that an evalution point x0 is feasible,
        or provide a cut that separates the feasible region and x0.

    Arguments:
        omega ([type]): perform assessment on x0
        S ([type]): Initial search space known to contain x*

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x: solution vector
        niter: number of iterations performed
    """
    for niter in range(1, options.max_iter):
        cut = omega.assess_feas(S.xc)  # query the oracle at S.xc
        if cut is None:  # feasible sol'n obtained
            return CInfo(True, niter, CutStatus.Success)

        cutstatus, tsq = S.update(cut)  # update S
        if cutstatus != CutStatus.Success:
            return CInfo(False, niter, cutstatus)

        if tsq < options.tol:
            return CInfo(False, niter, CutStatus.SmallEnough)
    return CInfo(False, options.max_iter, CutStatus.NoSoln)


def cutting_plane_optim(omega, S, t, options=Options()
) -> Tuple[Any, Any, int, CutStatus]:
    """Cutting-plane method for solving convex optimization problem

    Arguments:
        omega ([type]): perform assessment on x0
        S ([type]): Search Space containing x*
        t (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (Any): solution vector
        t: final best-so-far value
        ret {CInfo}
    """
    x_best = None
    status = CutStatus.Unknown

    for niter in range(1, options.max_iter):
        cut, t1 = omega.assess_optim(S.xc, t)
        if t1 is not None:  # better t obtained
            t = t1
            x_best = S.xc.copy()
        status, tsq = S.update(cut)
        if status != CutStatus.Success:
            return x_best, t, niter, status
        if tsq < options.tol:
            return x_best, t, niter, CutStatus.SmallEnough

    return x_best, t, options.max_iter, status


def cutting_plane_q(omega, S, t, options=Options()):
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        omega ([type]): perform assessment on x0
        S ([type]): Search Space containing x*
        t (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (float): solution vector
        t (float): best-so-far optimal value
        niter ([type]): number of iterations performed
    """
    # x_last = S.xc
    x_best = None
    status = CutStatus.Unknown
    retry = False
    for niter in range(1, options.max_iter):
        # retry = status == CutStatus.NoEffect
        cut, x0, t1, more_alt = omega.assess_optim_q(S.xc, t, retry)
        if t1 is not None:  # better t obtained
            t = t1
            x_best = x0.copy()
        status, tsq = S.update(cut)
        if status == CutStatus.NoEffect:
            if not more_alt:  # no more alternative cut
                return x_best, t, niter, status
            retry = True
        elif status == CutStatus.NoSoln:
            return x_best, t, niter, status
        if tsq < options.tol:
            return x_best, t, niter, CutStatus.SmallEnough

    return x_best, t, options.max_iter, status


def bsearch(omega, Interval: Tuple, options=Options()
) -> Tuple[Any, CInfo]:
    """[summary]

    Arguments:
        omega ([type]): [description]
        I ([type]): interval (initial search space)

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        [type]: [description]
    """
    # assume monotone
    # feasible = False
    lower, upper = Interval
    T = type(upper)  # T could be `int` or `Fraction`
    u_orig = upper
    status = CutStatus.Unknown

    for niter in range(1, options.max_iter):
        tau = (upper - lower) / 2
        if tau < options.tol:
            status = CutStatus.SmallEnough
            break
        t = T(lower + tau)
        if omega.assess_bs(t):  # feasible sol'n obtained
            upper = t
        else:
            lower = t

    ret = CInfo(upper != u_orig, niter + 1, status)
    return upper, ret


class bsearch_adaptor:
    def __init__(self, P, S, options=Options()):
        """[summary]

        Arguments:
            P ([type]): [description]
            S ([type]): [description]

        Keyword Arguments:
            options ([type]): [description] (default: {Options()})
        """
        self.P = P
        self.S = S
        self.options = options

    @property
    def x_best(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.S.xc

    def assess_bs(self, t):
        """[summary]

        Arguments:
            t (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        S = self.S.copy()
        self.P.update(t)
        ell_info = cutting_plane_feas(self.P, S, self.options)
        if ell_info.feasible:
            self.S.xc = S.xc
        return ell_info.feasible
