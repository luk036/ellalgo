from typing import Optional, Tuple, TypeVar

import numpy as np

from ellalgo.cutting_plane import OracleOptim

Arr = TypeVar("Arr", bound="np.ndarray")
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

    def __init__(self, params: Tuple[float, float, float], a: Arr, v: Arr):
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): p, A, k
            a (Arr): the output elasticities
            v (Arr): output price
        """
        p, A, k = params
        self.log_pA = np.log(p * A)
        self.log_k = np.log(k)
        self.v = v
        self.a = a

    def assess_optim(self, y: Arr, t: float) -> Tuple[Cut, Optional[float]]:
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

        log_Cobb = self.log_pA + self.a @ y
        q = self.v * np.exp(y)
        vx = q[0] + q[1]
        if (fj := np.log(t + vx) - log_Cobb) >= 0.0:
            g = q / (t + vx) - self.a
            return (g, fj), None

        t = np.exp(log_Cobb) - vx
        g = q / (t + vx) - self.a
        return (g, 0.0), t


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
        a: Arr,
        v: Arr,
        vparams: Tuple[float, float, float, float, float],
    ):
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): p, A, k
            a (Arr): the output elasticities
            v (Arr): output price
            vparams (Tuple): paramters for uncertainty
        """
        e1, e2, e3, e4, e5 = vparams
        self.a = a
        self.e = [e1, e2]
        p, A, k = params
        params_rb = p - e3, A, k - e4
        self.P = ProfitOracle(params_rb, a, v + e5)

    def assess_optim(self, y: Arr, t: float) -> Tuple[Cut, Optional[float]]:
        """Make object callable for cutting_plane_optim()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value

        Returns:
            Tuple[Cut, float]: Cut and the updated best-so-far value

        See also:
            cutting_plane_optim
        """
        a_rb = self.a.copy()
        for i in [0, 1]:
            a_rb[i] += -self.e[i] if y[i] > 0.0 else self.e[i]
        self.P.a = a_rb
        return self.P.assess_optim(y, t)


class ProfitQOracle:
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

    yd = None

    def __init__(self, params, a, v):
        """[summary]

        Arguments:
            params (Tuple[float, float, float]): p, A, k
            a (Arr): the output elasticities
            v (Arr): output price
        """
        self.P = ProfitOracle(params, a, v)

    def assess_optim_q(
        self, y: Arr, t: float, retry: bool
    ) -> Tuple[Cut, Arr, Optional[float], bool]:
        """Make object callable for cutting_plane_q()

        Arguments:
            y (Arr): input quantity (in log scale)
            t (float): the best-so-far optimal value
            retry ([type]): unused

        Raises:
            AssertionError: [description]

        Returns:
            Tuple: Cut, t, and the actual evaluation point

        See also:
            cutting_plane_q
        """
        if not retry:
            x = np.round(np.exp(y))
            if x[0] == 0:
                x[0] = 1
            if x[1] == 0:
                x[1] = 1
            self.yd = np.log(x)

        (g, h), t = self.P.assess_optim(self.yd, t)
        h += g @ (self.yd - y)
        return (g, h), self.yd, t, not retry
