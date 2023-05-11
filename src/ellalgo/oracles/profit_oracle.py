from typing import Optional, Tuple
from ellalgo.cutting_plane import OracleOptim, OracleOptimQ

import numpy as np

Arr = np.ndarray
Cut = Tuple[Arr, float]


class ProfitOracle(OracleOptim):
    """Oracle for a profit maximization problem.

    This example is taken from [Aliabadi and Salahi, 2013]

        max     p(A x1^α x2^β) − v1*x1 − v2*x2
        s.t.    x1 ≤ k

    where:

        p(A x1^α x2^β): Cobb-Douglas production function
        p: the market price per unit
        A: the scale of production
        α, β: the output elasticities
        x: input quantity
        v: output price
        k: a given constant that restricts the quantity of x1
    """

    def __init__(self, params: Tuple[float, float, float], elasticities: Arr, price_out: Arr) -> None:
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): price_per_unit, scale, limit
            elasticities (Arr): the output elasticities
            price_out (Arr): output price
        """
        price_per_unit, scale, limit = params
        self.log_pA = np.log(price_per_unit * scale)
        self.log_k = np.log(limit)
        self.price_out = price_out
        self.elasticities = elasticities

    def assess_optim(self, y: Arr, tea: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_optim()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_optim
        """
        if (fj := y[0] - self.log_k) > 0.0:  # constraint
            g = np.array([1.0, 0.0])
            return (g, fj), None

        log_Cobb = self.log_pA + self.elasticities @ y
        q = self.price_out * np.exp(y)
        vx = q[0] + q[1]
        if (fj := np.log(tea + vx) - log_Cobb) >= 0.0:
            g = q / (tea + vx) - self.elasticities
            return (g, fj), None

        tea = np.exp(log_Cobb) - vx
        g = q / (tea + vx) - self.elasticities
        return (g, 0.0), tea


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

        Arguments:
            params (Tuple[float, float, float]): price_per_unit, A, limit
            elasticities (Arr): the output elasticities
            price_out (Arr): output price
            vparams (Tuple): parameters for uncertainty
        """
        e1, e2, e3, e4, e5 = vparams
        self.elasticities = elasticities
        self.e = [e1, e2]
        price_per_unit, scale, limit = params
        params_rb = price_per_unit - e3, scale, limit - e4
        self.omega = ProfitOracle(params_rb, elasticities, price_out + np.array([e5, e5]))

    def assess_optim(self, y: Arr, tea: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_optim()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_optim
        """
        a_rb = self.elasticities.copy()
        for i in [0, 1]:
            a_rb[i] += -self.e[i] if y[i] > 0.0 else self.e[i]
        self.omega.elasticities = a_rb
        return self.omega.assess_optim(y, tea)


class ProfitQOracle(OracleOptimQ):
    """Oracle for a decrete profit maximization problem.

        max     p(A x1^α x2^β) - v1*x1 - v2*x2
        s.t.    x1 ≤ k

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

        Arguments:
            params (Tuple[float, float, float]): price_per_unit, scale, limit
            elasticities (Arr): the output elasticities
            price_out (Arr): output price
        """
        self.omega = ProfitOracle(params, elasticities, price_out)
        self.yd = np.array([0.0, 0.0])

    def assess_optim_q(
        self, y: Arr, tea: float, retry: bool
    ) -> Tuple[Cut, Arr, Optional[float], bool]:
        """Make object callable for cutting_plane_optim_q()

        Arguments:
            y (Arr): input quantity (in log scale)
            tea (float): the best-so-far optimal value
            retry ([type]): unused

        Raises:
            AssertionError: [description]

        Returns:
            Tuple: Cut, tea, and the actual evaluation point

        See also:
            cutting_plane_optim_q
        """
        if not retry:
            x = np.round(np.exp(y))
            if x[0] == 0:
                x[0] = 1.0 # nearest integer than 0
            if x[1] == 0:
                x[1] = 1.0
            self.yd = np.log(x)

        (g, h), tnew = self.omega.assess_optim(self.yd, tea)
        h += g.dot(self.yd - y)
        return (g, h), self.yd, tnew, False
