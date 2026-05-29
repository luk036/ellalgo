from enum import Enum


class CutStatus(Enum):
    Success = 0
    NoSoln = 1
    NoEffect = 2
    Unknown = 3


class Options:
    max_iters: int = 2000
    tolerance: float = 1e-20
    verbose: bool = False
