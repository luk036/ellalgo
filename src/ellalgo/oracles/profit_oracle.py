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

    log_pA: float
    log_k: float
    price_out: Arr
    elasticities: Arr

    def __init__(
        self, params: Tuple[float, float, float], elasticities: Arr, price_out: Arr
    ) -> None:
        """[summary]

        Args:
            params (Tuple[float, float, float]): unit_price, scale, limit
            elasticities (Arr): the output elasticities
            price_out (Arr): output price

        Examples:
            >>> oracle = ProfitOracle((0.1, 1.0, 10.0), np.array([0.1, 0.2]), np.array([1.0, 2.0]))
        """
        unit_price, scale, limit = params
        self.log_pA = math.log(unit_price * scale)
        self.log_k = math.log(limit)
        self.price_out = price_out
        self.elasticities = elasticities

    def assess_optim(self, y: Arr, target: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_optim()

        Args:
            y (Arr): input quantity (in log scale)
            target (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_optim
        """
        if (fj := y[0] - self.log_k) > 0.0:  # constraint
            g = np.array([1.0, 0.0])
            return (g, fj), None

        log_Cobb = self.log_pA + self.elasticities.dot(y)
        q = self.price_out * np.exp(y)
        vx = q[0] + q[1]
        if (fj := math.log(target + vx) - log_Cobb) >= 0.0:
            g = q / (target + vx) - self.elasticities
            return (g, fj), None

        target = np.exp(log_Cobb) - vx
        g = q / (target + vx) - self.elasticities
        return (g, 0.0), target


class ProfitRbOracle(OracleOptim):
    """Oracle for a robust profit maximization problem.

    This example is taken from [Aliabadi and Salahi, 2013]:

      max  p'(A x1^α' x2^β') - v1'*x1 - v2'*x2
      s.t. x1 ≤ k'

    where:
        α' = α ± e1
        β' = β ± e2
        p' = p ± e3
        k' = k ± e4
        v' = v ± e5

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
        """[summary]

        Args:
            params (Tuple[float, float, float]): unit_price, scale, limit
            elasticities (Arr): the output elasticities
            price_out (Arr): output price
            vparams (Tuple): parameters for uncertainty
        """
        e1, e2, e3, e4, e5 = vparams
        self.elasticities = elasticities
        self.e = [e1, e2]
        unit_price, scale, limit = params
        params_rb = unit_price - e3, scale, limit - e4
        self.omega = ProfitOracle(
            params_rb, elasticities, price_out + np.array([e5, e5])
        )

    def assess_optim(self, y: Arr, target: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_optim()

        Args:
            y (Arr): input quantity (in log scale)
            target (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_optim
        """
        a_rb = copy.copy(self.elasticities)
        for i in [0, 1]:
            a_rb[i] += -self.e[i] if y[i] > 0.0 else self.e[i]
        self.omega.elasticities = a_rb
        return self.omega.assess_optim(y, target)


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
        """[summary]

        Args:
            params (Tuple[float, float, float]): unit_price, scale, limit
            elasticities (Arr): the output elasticities
            price_out (Arr): output price
        """
        self.omega = ProfitOracle(params, elasticities, price_out)
        self.yd = np.array([0.0, 0.0])

    def assess_optim_q(
        self, y: Arr, target: float, retry: bool
    ) -> Tuple[Cut, Arr, Optional[float], bool]:
        """Make object callable for cutting_plane_optim_q()

        Args:
            y (Arr): input quantity (in log scale)
            target (float): the best-so-far optimal value
            retry ([type]): unused

        Raises:
            AssertionError: [description]

        Returns:
            Tuple: Cut, target, and the actual evaluation point

        See also:
            cutting_plane_optim_q
        """
        if not retry:
            x = np.round(np.exp(y))
            if x[0] == 0:
                x[0] = 1.0  # nearest integer than 0
            if x[1] == 0:
                x[1] = 1.0
            self.yd = np.log(x)

        (g, h), tnew = self.omega.assess_optim(self.yd, target)
        h += g.dot(self.yd - y)
        return (g, h), self.yd, tnew, False
