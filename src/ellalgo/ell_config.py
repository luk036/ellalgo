from enum import Enum


class CutStatus(Enum):
    Success = 0
    NoSoln = 1
    NoEffect = 2
    Unknown = 3


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

