from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union

from numpy import ndarray

ArrayType = Union[float, ndarray]  # one or multi dimensional
CutChoice = Union[float, ndarray]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]
FloatOrInt = Union[float, int]


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
    def __init__(self, feasible: bool, num_iters: int, status: CutStatus) -> None:
        """Construct a new CInfo object

        Arguments:
            feasible (bool): [description]
            num_iters (int): [description]
            status (int): [description]
        """
        self.feasible: bool = feasible
        self.num_iters: int = num_iters
        self.status: CutStatus = status


class OracleFeas(ABC):
    @abstractmethod
    def assess_feas(self, x: ArrayType) -> Optional[Cut]:
        pass


class OracleFeas2(OracleFeas):
    @abstractmethod
    def assess_feas(self, x: ArrayType) -> Optional[Cut]:
        pass

    @abstractmethod
    def update(self, t: FloatOrInt):
        pass


class OracleOptim(ABC):
    @abstractmethod
    def assess_optim(
        self, x: ndarray, t: float  # what?
    ) -> Tuple[Cut, Optional[float]]:
        pass


class OracleOptimQ(ABC):
    @abstractmethod
    def assess_optim_q(
        self, x: ndarray, t: float, retry: bool
    ) -> Tuple[Cut, ndarray, Optional[float], bool]:
        pass


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, t: FloatOrInt) -> bool:
        pass


class SearchSpace(ABC):
    @abstractmethod
    def update(self, cut: Cut) -> Tuple[CutStatus, float]:
        pass

    @abstractmethod
    def xc(self) -> ndarray:
        pass


"""
CuttingPlane -> SearchSpace: request xc
SearchSpace -> CuttingPlane: return xc
CuttingPlane -> OracleFeas: assess_feas(xc)
OracleFeas -> CuttingPlane: return cut
CuttingPlane -> SearchSpace: update by the cut
"""


class SearchSpace2(SearchSpace):
    @abstractmethod
    def copy(self) -> SearchSpace:
        pass

    @abstractmethod
    def set_xc(self, xc: ndarray) -> None:
        pass


def cutting_plane_feas(
    omega: OracleFeas, space: SearchSpace, options=Options()
) -> CInfo:
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

    Arguments:
        omega (OracleFeas): perform assessment on x0
        space (SearchSpace): Initial search space known to contain x*

    Keyword Arguments:
        options (Options): [description] (default: {Options()})

    Returns:
        feasible (bool): whether it is feasible
        niter (int): number of iterations performed
        cutStatus (CutStatus): cut status
    """
    for niter in range(options.max_iter):
        cut = omega.assess_feas(space.xc())  # query the oracle at S.xc()
        if cut is None:  # feasible sol'n obtained
            return CInfo(True, niter, CutStatus.Success)
        cutstatus, tsq = space.update(cut)  # update S
        if cutstatus != CutStatus.Success:
            return CInfo(False, niter, cutstatus)
        if tsq < options.tol:
            return CInfo(False, niter, CutStatus.SmallEnough)
    return CInfo(False, options.max_iter, CutStatus.NoSoln)


def cutting_plane_optim(
    omega: OracleOptim, space: SearchSpace, t: float, options=Options()
) -> Tuple[Optional[ndarray], float, int, CutStatus]:
    """Cutting-plane method for solving convex optimization problem

    Arguments:
        omega (OracleOptim): perform assessment on x0
        space (SearchSpace): Search Space containing x*
        t (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (Any): solution vector
        t: final best-so-far value
        ret {CInfo}
    """
    x_best = None
    for niter in range(options.max_iter):
        cut, t1 = omega.assess_optim(space.xc(), t)
        if t1 is not None:  # better t obtained
            t = t1
            x_best = space.xc().copy()
        status, tsq = space.update(cut)
        if status != CutStatus.Success:
            return x_best, t, niter, status
        if tsq < options.tol:
            return x_best, t, niter, CutStatus.SmallEnough
    return x_best, t, options.max_iter, CutStatus.Success


def cutting_plane_q(
    omega: OracleOptimQ, S, t: float, options=Options()
) -> Tuple[Optional[ArrayType], float, int, CutStatus]:
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        omega (OracleOptimQ): perform assessment on x0
        S ([type]): Search Space containing x*
        t (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (float): solution vector
        t (float): best-so-far optimal value
        niter ([type]): number of iterations performed
    """
    # x_last = S.xc()
    x_best = None
    retry = False
    for niter in range(options.max_iter):
        cut, x0, t1, more_alt = omega.assess_optim_q(S.xc(), t, retry)
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
    return x_best, t, options.max_iter, CutStatus.Success


def bsearch(
    omega: OracleBS, intrvl: Tuple[FloatOrInt, FloatOrInt], options=Options()
) -> Tuple[FloatOrInt, int, CutStatus]:
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
    lower, upper = intrvl
    T = type(upper)  # T could be `int`
    for niter in range(options.max_iter):
        tau = (upper - lower) / 2
        if tau < options.tol:
            return upper, niter, CutStatus.SmallEnough
        t = T(lower + tau)
        if omega.assess_bs(t):  # feasible sol'n obtained
            upper = t
        else:
            lower = t
    return upper, options.max_iter, CutStatus.Unknown


class bsearch_adaptor:
    def __init__(self, P: OracleFeas2, S: SearchSpace2, options=Options()) -> None:
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
    def x_best(self) -> ndarray:
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.S.xc()

    def assess_bs(self, t: FloatOrInt) -> bool:
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
            self.S.set_xc(S.xc())
        return ell_info.feasible
