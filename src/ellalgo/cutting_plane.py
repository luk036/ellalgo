from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from numpy import ndarray
else:
    from typing import Any
    ndarray = Any

ArrayType = Union[float, ndarray]  # one or multi dimensional
CutChoice = Union[float, ndarray]  # single or parallel
Cut = Tuple[ArrayType, CutChoice]
Num = Union[float, int]


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
    def __init__(self, feasible: bool, num_iters: int) -> None:
        """Construct a new CInfo object

        Arguments:
            feasible (bool): [description]
            num_iters (int): [description]
            status (int): [description]
        """
        self.feasible: bool = feasible
        self.num_iters: int = num_iters


class OracleFeas(ABC):
    @abstractmethod
    def assess_feas(self, x: ArrayType) -> Optional[Cut]:
        pass


class OracleFeas2(OracleFeas):
    @abstractmethod
    def update(self, tea: Num):
        pass


class OracleOptim(ABC):
    @abstractmethod
    def assess_optim(
        self, xc: ndarray, tea: float  # what?
    ) -> Tuple[Cut, Optional[float]]:
        pass


class OracleFeasQ(ABC):
    @abstractmethod
    def assess_feas_q(
        self, xc: ndarray, retry: bool
    ) -> Tuple[Optional[Cut], Optional[ndarray], bool]:
        pass


class OracleOptimQ(ABC):
    @abstractmethod
    def assess_optim_q(
        self, xc: ndarray, tea: float, retry: bool
    ) -> Tuple[Cut, ndarray, Optional[float], bool]:
        pass


class OracleBS(ABC):
    @abstractmethod
    def assess_bs(self, tea: Num) -> bool:
        pass


class SearchSpace(ABC):
    @abstractmethod
    def update(self, cut: Cut, central_cut: bool = False) -> CutStatus:
        pass

    @abstractmethod
    def xc(self) -> ndarray:
        pass

    @abstractmethod
    def tsq(self) -> float:
        pass


class SearchSpace2(SearchSpace):
    @abstractmethod
    def copy(self) -> SearchSpace:
        pass

    @abstractmethod
    def set_xc(self, xc: ndarray) -> None:
        pass


def cutting_plane_feas(
    omega: OracleFeas, space: SearchSpace, options=Options()
) -> Tuple[Optional[ndarray], int]:
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
        omega (OracleFeas): perform assessment on xinit
        space (SearchSpace): Initial search space known to contain x*

    Keyword Arguments:
        options (Options): [description] (default: {Options()})

    Returns:
        feasible (bool): whether it is feasible
        niter (int): number of iterations performed
        cutStatus (CutStatus): cut status
    """
    for niter in range(options.max_iter):
        cut = omega.assess_feas(space.xc())  # query the oracle at space.xc()
        if cut is None:  # feasible sol'n obtained
            return space.xc(), niter
        cutstatus = space.update(cut)  # update space
        if cutstatus != CutStatus.Success or space.tsq() < options.tol:
            return None, niter
    return None, options.max_iter


def cutting_plane_optim(
    omega: OracleOptim, space: SearchSpace, tea: float, options=Options()
) -> Tuple[Optional[ndarray], float, int]:
    """Cutting-plane method for solving convex optimization problem

    Arguments:
        omega (OracleOptim): perform assessment on xinit
        space (SearchSpace): Search Space containing x*
        tea (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (Any): solution vector
        tea: final best-so-far value
        ret {CInfo}
    """
    x_best = None
    for niter in range(options.max_iter):
        cut, tea1 = omega.assess_optim(space.xc(), tea)
        if tea1 is not None:  # better t obtained
            tea = tea1
            x_best = space.xc().copy()
            status = space.update(cut, central_cut=True)
        else:
            status = space.update(cut, central_cut=False)
        if status != CutStatus.Success or space.tsq() < options.tol:
            return x_best, tea, niter
    return x_best, tea, options.max_iter


def cutting_plane_feas_q(
        omega: OracleFeasQ, space: SearchSpace, options=Options()
) -> Tuple[Optional[ArrayType], int]:
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        omega (OracleFeasQ): perform assessment on xinit
        space ([type]): Search Space containing x*

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (float): solution vector
        niter ([type]): number of iterations performed
    """
    # x_last = space.xc()
    retry = False
    for niter in range(options.max_iter):
        cut, x_q, more_alt = omega.assess_feas_q(space.xc(), retry)
        if cut is None:  # better t obtained
            return x_q, niter
        cutstatus = space.update(cut)
        if cutstatus == CutStatus.NoEffect:
            if not more_alt:  # no more alternative cut
                return None, niter
            retry = True
        elif cutstatus == CutStatus.NoSoln:
            return None, niter
        if space.tsq() < options.tol:
            return None, niter
    return None, options.max_iter


def cutting_plane_q(
        omega: OracleOptimQ, space: SearchSpace, tea: float, options=Options()
) -> Tuple[Optional[ArrayType], float, int]:
    """Cutting-plane method for solving convex discrete optimization problem

    Arguments:
        omega (OracleOptimQ): perform assessment on xinit
        space ([type]): Search Space containing x*
        tea (float): initial best-so-far value

    Keyword Arguments:
        options ([type]): [description] (default: {Options()})

    Returns:
        x_best (float): solution vector
        tea (float): best-so-far optimal value
        niter ([type]): number of iterations performed
    """
    # x_last = space.xc()
    x_best = None
    retry = False
    for niter in range(options.max_iter):
        cut, x_q, t1, more_alt = omega.assess_optim_q(space.xc(), tea, retry)
        if t1 is not None:  # better t obtained
            tea = t1
            x_best = x_q.copy()
        status = space.update(cut)
        if status == CutStatus.NoEffect:
            if not more_alt:  # no more alternative cut
                return x_best, tea, niter
            retry = True
        elif status == CutStatus.NoSoln:
            return x_best, tea, niter
        if space.tsq() < options.tol:
            return x_best, tea, niter
    return x_best, tea, options.max_iter


def bsearch(
    omega: OracleBS, intrvl: Tuple[Num, Num], options=Options()
) -> Tuple[Num, int]:
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
            return upper, niter
        tea = T(lower + tau)
        if omega.assess_bs(tea):  # feasible sol'n obtained
            upper = tea
        else:
            lower = tea
    return upper, options.max_iter


class bsearch_adaptor:
    def __init__(
        self, omega: OracleFeas2, space: SearchSpace2, options=Options()
    ) -> None:
        """[summary]

        Arguments:
            omega ([type]): [description]
            space ([type]): [description]

        Keyword Arguments:
            options ([type]): [description] (default: {Options()})
        """
        self.omega = omega
        self.space = space
        self.options = options

    @property
    def x_best(self) -> ndarray:
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.space.xc()

    def assess_bs(self, tea: Num) -> bool:
        """[summary]

        Arguments:
            t (float): the best-so-far optimal value

        Returns:
            [type]: [description]
        """
        space = self.space.copy()
        self.omega.update(tea)
        x_feas, _ = cutting_plane_feas(self.omega, space, self.options)
        if x_feas is not None:
            self.space.set_xc(x_feas)
            return True
        return False
