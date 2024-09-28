"""
Profit Oracle

This code defines several classes that implement oracles for profit maximization problems. An oracle, in this context, is a function that helps solve optimization problems by providing information about the feasibility and optimality of potential solutions.

The main class, ProfitOracle, is designed to solve a specific type of profit maximization problem. It takes as input parameters related to production (like price, scale, and limits) and output elasticities. The goal is to find the optimal input quantities that maximize profit, given certain constraints.

The ProfitOracle class has methods to assess the feasibility of a solution (assess_feas) and to find the optimal solution (assess_optim). These methods take as input a vector y (representing input quantities in log scale) and a gamma value (representing the current best solution). They output "cuts", which are linear constraints that help narrow down the search for the optimal solution.

The code also includes two variations of the profit oracle:

1. ProfitRbOracle: This is a robust version of the profit oracle that can handle some uncertainty in the input parameters.

2. ProfitQOracle: This version deals with discrete (integer) input quantities, as opposed to continuous ones.

The main logic flow in these classes involves calculating various economic functions (like Cobb-Douglas production functions) and their gradients. The code uses these calculations to determine if a given solution is feasible and to guide the search towards the optimal solution.

The output of these oracles is typically a "cut" (a linear constraint) and possibly an updated best solution (gamma). These outputs are used by an external optimization algorithm (not shown in this code) to iteratively improve the solution until the optimal one is found.

For beginners, it's important to understand that this code is implementing mathematical optimization techniques. While the details might be complex, the basic idea is to efficiently search for the best solution to a profit maximization problem, given certain constraints and economic relationships.
"""

import copy
import math
from typing import Optional, Tuple

import numpy as np

from ellalgo.cutting_plane import OracleOptim, OracleOptimQ

Arr = np.ndarray
Cut = Tuple[Arr, float]


class ProfitOracle(OracleOptim):
    """Oracle for a profit maximization problem.

    This example is taken from [Aliabadi and Salahi, 2013]

      max  p(A x1^α x2^β) − v1*x1 − v2*x2
      s.t. x1 ≤ k

    where:

      p(A x1^α x2^β): Cobb-Douglas production function
      p: the market price per unit
      A: the scale of production
      α, β: the output elasticities
      x: input quantity
      v: output price
      k: a given constant that restricts the quantity of x1
    """

    idx: int = -1  # for round robin
    log_Cobb: float
    q: Arr
    vx: float

    log_pA: float
    log_k: float
    price_out: Arr
    elasticities: Arr

    def __init__(
        self, params: Tuple[float, float, float], elasticities: Arr, price_out: Arr
    ) -> None:
        """
        The function initializes a ProfitOracle object with given parameters.

        :param params: The `params` parameter is a tuple containing three float values: `unit_price`,
            `scale`, and `limit`. These values are used to calculate the logarithm of the unit price (`log_pA`)
            and the logarithm of the limit (`log_k`)

        :type params: Tuple[float, float, float]

        :param elasticities: The `elasticities` parameter is an array that represents the output
            elasticities. It contains the elasticity values for each output

        :type elasticities: Arr

        :param price_out: The `price_out` parameter is an array that represents the output prices

        :type price_out: Arr

        Examples:
            >>> oracle = ProfitOracle((0.1, 1.0, 10.0), np.array([0.1, 0.2]), np.array([1.0, 2.0]))
        """
        unit_price, scale, limit = params
        self.log_pA = math.log(unit_price * scale)
        self.log_k = math.log(limit)
        self.price_out = price_out
        self.elasticities = elasticities
        self.fns = (self.fn1, self.fn2)
        self.grads = (self.grad1, self.grad2)

    def fn1(self, y: Arr, _: float) -> float:
        return y[0] - self.log_k  # constraint

    def fn2(self, y: Arr, gamma: float) -> float:
        self.log_Cobb = self.log_pA + self.elasticities.dot(y)
        self.q = self.price_out * np.exp(y)
        self.vx = self.q[0] + self.q[1]
        return math.log(gamma + self.vx) - self.log_Cobb

    def grad1(self, _: float) -> Arr:
        return np.array([1.0, 0.0])

    def grad2(self, gamma: float) -> Arr:
        return self.q / (gamma + self.vx) - self.elasticities

    def assess_feas(self, y: Arr, gamma: float) -> Optional[Cut]:
        """
        The `assess_feas` function takes in an input quantity `y` and a gamma value, and an optional Cut.

        :param y: The parameter `y` is an array representing the input quantity in log scale

        :type y: Arr

        :param gamma: The `gamma` parameter is the best-so-far optimal value. It represents the gamma
            value that the optimization algorithm is trying to achieve or improve upon

        :type gamma: float

        :return: The function `assess_feas` returns an optional Cut. The `Cut` object represents a
            linear constraint in the form of a tuple `(grad, fj)`, where `grad`
            is a numpy array representing the coefficients of the linear constraint and `fj` is a float
            representing the right-hand side of the constraint.

        See also:
            cutting_plane_optim
        """
        for _ in [0, 1]:
            self.idx += 1
            if self.idx == 2:
                self.idx = 0  # round robin
            if (fj := self.fns[self.idx](y, gamma)) > 0:
                return self.grads[self.idx](gamma), fj
        return None

    def assess_optim(self, y: Arr, gamma: float) -> Tuple[Cut, Optional[float]]:
        """
        The `assess_optim` function takes in an input quantity `y` and a gamma value, and returns a tuple
        containing a cut and an updated best-so-far value.

        :param y: The parameter `y` is an array representing the input quantity in log scale

        :type y: Arr

        :param gamma: The `gamma` parameter is the best-so-far optimal value. It represents the gamma
            value that the optimization algorithm is trying to achieve or improve upon

        :type gamma: float

        :return: The function `assess_optim` returns a tuple containing a `Cut` object and an optional float
            value. The `Cut` object represents a linear constraint in the form of a tuple `(g, fj)`, where `g`
            is a numpy array representing the coefficients of the linear constraint and `fj` is a float
            representing the right-hand side of the constraint.

        See also:
            cutting_plane_optim
        """
        cut = self.assess_feas(y, gamma)
        if cut is not None:
            return cut, None
        gamma = np.exp(self.log_Cobb) - self.vx
        grad = self.q / (gamma + self.vx) - self.elasticities
        return (grad, 0.0), gamma


class ProfitRbOracle(OracleOptim):
    """Oracle for a robust profit maximization problem.

    This example is taken from [Aliabadi and Salahi, 2013]:

    See also:
        ProfitOracle
    """

    def __init__(
        self,
        params: Tuple[float, float, float],
        elasticities: Arr,
        price_out: Arr,
        vparams: Tuple[float, float, float, float, float],
    ) -> None:
        """
        The function initializes an object with given parameters and calculates the omega value using a
        ProfitOracle object.

        :param params: The `params` parameter is a tuple of three floats: `unit_price`, `scale`, and
            `limit`. These parameters are used to calculate `params_rb` in the code

        :type params: Tuple[float, float, float]

        :param elasticities: The elasticities parameter is a numpy array that represents the output
            elasticities. It is used in the ProfitOracle function

        :type elasticities: Arr

        :param price_out: The `price_out` parameter is an array representing the output price

        :type price_out: Arr

        :param vparams: The `vparams` parameter is a tuple containing five float values: `e1`, `e2`, `e3`,
            `e4`, and `e5`

        :type vparams: Tuple[float, float, float, float, float]
        """
        e1, e2, e3, e4, e5 = vparams
        self.elasticities = elasticities
        self.uie = [e1, e2]
        unit_price, scale, limit = params
        params_rb = unit_price - e3, scale, limit - e4
        self.omega = ProfitOracle(
            params_rb, elasticities, price_out + np.array([e5, e5])
        )

    def assess_optim(self, y: Arr, gamma: float) -> Tuple[Cut, Optional[float]]:
        """
        The `assess_optim` function takes in an input quantity `y` and a gamma value, and returns a tuple
        containing a cut and an updated best-so-far value.

        :param y: The parameter `y` is an array representing the input quantity in log scale

        :type y: Arr

        :param gamma: The `gamma` parameter is the best-so-far optimal value. It represents the current
            best value that has been achieved in the optimization process

        :type gamma: float

        :return: The function `assess_optim` returns a tuple containing a `Cut` object and an optional float
            value.

        See also:
            cutting_plane_optim
        """
        a_rb = copy.copy(self.elasticities)
        for i in [0, 1]:
            a_rb[i] += -self.uie[i] if y[i] > 0.0 else self.uie[i]
        self.omega.elasticities = a_rb
        return self.omega.assess_optim(y, gamma)


class ProfitQOracle(OracleOptimQ):
    """Oracle for a discrete profit maximization problem.

      max   p(A x1^α x2^β) - v1*x1 - v2*x2
      s.t.  x1 ≤ k

    where:

        p(A x1^α x2^β): Cobb-Douglas production function
        p: the market price per unit
        A: the scale of production
        α, β: the output elasticities
        x: input quantity (must be integer value)
        v: output price
        k: a given constant that restricts the quantity of x1

    Raises:
        AssertionError: [description]

    See also:
        ProfitOracle
    """

    yd: np.ndarray

    def __init__(self, params, elasticities, price_out) -> None:
        """
        The function initializes an instance of a class with given parameters and arrays.

        :param params: The `params` parameter is a tuple containing three float values: `unit_price`,
            `scale`, and `limit`

        :param elasticities: The elasticities parameter is an array that represents the output elasticities.
            It contains the elasticity values for each output

        :param price_out: The `price_out` parameter is an array that represents the output prices of the
            goods or services. It contains the prices of different outputs
        """
        self.omega = ProfitOracle(params, elasticities, price_out)
        self.yd = np.array([0.0, 0.0])

    def assess_optim_q(
        self, y: Arr, gamma: float, retry: bool
    ) -> Tuple[Cut, Arr, Optional[float], bool]:
        """
        The `assess_optim_q` function takes in an input quantity `y` in log scale, a gamma value, and a
        retry flag, and returns a tuple containing a cut, the gamma value, and the evaluation point.

        :param y: An array representing the input quantity in log scale

        :type y: Arr

        :param gamma: The `gamma` parameter is the best-so-far optimal value. It represents the current
            best value that the optimization algorithm has found

        :type gamma: float

        :param retry: A boolean flag indicating whether the optimization should be retried or not

        :type retry: bool

        :return: The function `assess_optim_q` returns a tuple containing the following elements:

        See also:
            cutting_plane_optim_q
        """
        if not retry:
            # try continous y first
            if cut := self.omega.assess_feas(y, gamma):
                return cut, y, None, True

            xd = np.round(np.exp(y))
            if xd[0] == 0:
                xd[0] = 1.0  # nearest integer than 0
            if xd[1] == 0:
                xd[1] = 1.0
            self.yd = np.log(xd)

        (grad, beta), gamma_new = self.omega.assess_optim(self.yd, gamma)
        beta += grad.dot(self.yd - y)  # reference as y
        return (grad, beta), self.yd, gamma_new, not retry
