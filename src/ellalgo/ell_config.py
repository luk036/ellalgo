from enum import Enum


# The above class defines an enumeration for different cut statuses.
class CutStatus(Enum):
    Success = 0
    NoSoln = 1
    NoEffect = 2
    Unknown = 3


# The class "Options" defines two attributes, "max_iters" and "tol", with default values of 2000 and
# 1e-8 respectively.
class Options:
    max_iters: int = 2000  # maximum number of iterations
    tol: float = 1e-8  # error tolerance


# The `CInfo` class represents information about a computation, including whether it is feasible and
# the number of iterations it took.
class CInfo:
    def __init__(self, feasible: bool, num_iters: int) -> None:
        """
        The function initializes a new CInfo object with the given feasibility and number of iterations.

        :param feasible: A boolean value indicating whether the solution is feasible or not
        :type feasible: bool
        :param num_iters: The `num_iters` parameter represents the number of iterations or steps taken
        in a process or algorithm. It is an integer value that indicates how many times a certain
        operation or calculation has been performed
        :type num_iters: int
        """
        self.feasible: bool = feasible
        self.num_iters: int = num_iters
