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
    """Oracle for a profit maximization problem using Cobb-Douglas production function.

    This implementation follows the formulation from [Aliabadi and Salahi, 2013]:

    Optimization problem:
      max  p(A y₁^α y₂^β) − v₁y₁ − v₂y₂
      s.t. y₁ ≤ k

    Where:
    - p(A y₁^α y₂^β): Cobb-Douglas production function in exponential form
    - p: Market price per unit
    - A: Production scale factor
    - α, β: Output elasticities (α + β ≤ 1 for constant returns to scale)
    - y: Input quantities (decision variables in log scale)
    - v: Input prices
    - k: Upper bound constraint for x₁

    The oracle uses cutting plane methods to iteratively refine the solution space.
    """

    idx: int = -1  # Index for round-robin constraint checking
    log_Cobb: float  # Log value of Cobb-Douglas production
    q: Arr  # Intermediate calculation of price_out * exp(y)
    vy: float  # Total variable cost v₁y₁ + v₂y₂

    log_pA: float  # log(p*A) precomputed value
    log_k: float  # log(k) constraint value
    price_out: Arr  # Output prices [v₁, v₂]
    elasticities: Arr  # Elasticity parameters [α, β]

    def __init__(
        self, params: Tuple[float, float, float], elasticities: Arr, price_out: Arr
    ) -> None:
        """Initialize profit maximization oracle with problem parameters.

        Parameters:
        :param params: Tuple containing:
            - unit_price (p): Price per output unit
            - scale (A): Production scale factor
            - limit (k): Upper bound for x₁
        :param elasticities: Array [α, β] of output elasticities
        :param price_out: Array [v₁, v₂] of input prices

        Mathematical precomputations:
        - log_pA = log(p*A) simplifies subsequent exponential calculations
        - log_k = log(k) enables log-space constraint checking
        """
        unit_price, scale, limit = params
        self.log_pA = math.log(unit_price * scale)
        self.log_k = math.log(limit)
        self.price_out = price_out
        self.elasticities = elasticities
        self.fns = (self.fn1, self.fn2)  # Constraint functions
        self.grads = (self.grad1, self.grad2)  # Gradient functions

    def fn1(self, x: Arr, _: float) -> float:
        """Constraint function for y₁ ≤ k (in log-space).

        Args:
            x: Log-scale input vector [log(y₁), log(y₂)]

        Returns:
            Constraint violation measure: x[0] - log(k)
            Positive values indicate constraint violation
        """
        return x[0] - self.log_k  # log(y₁) ≤ log(k) → y₁ ≤ k

    def fn2(self, x: Arr, gamma: float) -> float:
        """Optimality condition function for profit maximization.

        Computes:
        - Cobb-Douglas value in log-space: log(pA) + αlog(y₁) + βlog(y₂)
        - Variable costs: v₁y₁ + v₂y₂
        - Optimality gap: log(γ + vy) - log_Cobb

        Args:
            x: Log-scale input vector
            gamma: Current best profit estimate

        Updates intermediate values used in gradient calculations
        """
        self.log_Cobb = self.log_pA + self.elasticities.dot(x)
        self.q = self.price_out * np.exp(x)  # v₁y₁, v₂y₂
        self.vy = self.q[0] + self.q[1]  # Total variable cost
        return math.log(gamma + self.vy) - self.log_Cobb

    def grad1(self, _: float) -> Arr:
        """Gradient for y₁ ≤ k constraint.

        Returns:
            Gradient vector [1, 0] since ∂(x₀ - log_k)/∂x = (1, 0)
        """
        return np.array([1.0, 0.0])

    def grad2(self, gamma: float) -> Arr:
        """Gradient of optimality condition function.

        Computes:
            ∇f = [v₁y₁/(γ+vy) - α, v₂y₂/(γ+vy) - β]

        Args:
            gamma: Current profit estimate used in denominator

        Uses precomputed q (v₁y₁, v₂y₂) from last fn2 call
        """
        return self.q / (gamma + self.vy) - self.elasticities

    def assess_feas(self, xc: Arr, gamma: float) -> Optional[Cut]:
        """Feasibility assessment using round-robin constraint checking.

        Implements:
        - Alternates between checking y₁ constraint (fn1) and optimality (fn2)
        - Returns first violated constraint found

        Args:
            xc: Current solution point in log-space
            gamma: Current best profit estimate

        Returns:
            Cut (gradient, violation) if constraint violated
            None if all constraints satisfied
        """
        for _ in [0, 1]:
            self.idx += 1
            if self.idx == 2:
                self.idx = 0  # Round-robin reset
            if (fj := self.fns[self.idx](xc, gamma)) > 0:
                return self.grads[self.idx](gamma), fj
        return None

    def assess_optim(self, xc: Arr, gamma: float) -> Tuple[Cut, Optional[float]]:
        """Optimization assessment generating optimality cuts.

        Workflow:
        1. Check feasibility using assess_feas
        2. If feasible, calculate new profit estimate and optimality cut
        3. If infeasible, return feasibility cut

        Args:
            y: Proposed solution point
            gamma: Current best profit value

        Returns:
            Tuple containing:
            - Cut (gradient, beta)
            - Updated gamma (None if infeasible)
        """
        cut = self.assess_feas(xc, gamma)
        if cut is not None:
            return cut, None
        # Calculate new profit estimate: pA x^α - vy
        gamma = np.exp(self.log_Cobb) - self.vy
        grad = self.q / (gamma + self.vy) - self.elasticities
        return (grad, 0.0), gamma


class ProfitRbOracle(OracleOptim):
    """Robust profit oracle handling parameter uncertainty.

    Implements robust optimization version from [Aliabadi and Salahi, 2013]
    considering uncertainties in:
    - Elasticity parameters (α, β)
    - Price parameters (p, v)
    - Production limit (k)

    Uses interval-based uncertainty sets for robust constraint satisfaction.
    """

    def __init__(
        self,
        params: Tuple[float, float, float],
        elasticities: Arr,
        price_out: Arr,
        vparams: Tuple[float, float, float, float, float],
    ) -> None:
        """Initialize robust oracle with uncertainty parameters.

        Parameters:
        :param vparams: Uncertainty parameters tuple (ε₁, ε₂, ε₃, ε₄, ε₅) representing:
            - ε₁, ε₂: Elasticity uncertainties
            - ε₃: Price uncertainty
            - ε₄: Production limit uncertainty
            - ε₅: Input price uncertainty

        Constructs worst-case scenario parameters for robust optimization.
        """
        e1, e2, e3, e4, e5 = vparams
        self.elasticities = elasticities
        self.uie = [e1, e2]  # Elasticity uncertainties
        unit_price, scale, limit = params
        # Construct robust parameters:
        params_rb = (
            unit_price - e3,  # Worst-case price decrease
            scale,
            limit - e4,  # Worst-case capacity reduction
        )
        self.omega = ProfitOracle(
            params_rb,
            elasticities,
            price_out + np.array([e5, e5]),  # Worst-case input price increase
        )

    def assess_optim(self, xc: Arr, gamma: float) -> Tuple[Cut, Optional[float]]:
        """Robust optimization assessment accounting for parameter uncertainties.

        Adjusts elasticities based on direction of uncertainty impact:
        - Decreases effective α, β when y > 0 (conservative adjustment)
        - Increases effective α, β when y ≤ 0 (worst-case scenario)
        """
        a_rb = copy.copy(self.elasticities)
        for i in [0, 1]:
            a_rb[i] += -self.uie[i] if xc[i] > 0.0 else self.uie[i]
        self.omega.elasticities = a_rb
        return self.omega.assess_optim(xc, gamma)


class ProfitQOracle(OracleOptimQ):
    """Discrete profit oracle for integer input quantities.

    Solves mixed-integer version of the profit maximization problem:
      max  p(A y₁^α y₂^β) − v₁y₁ − v₂y₂
      s.t. x₁ ≤ k, x ∈ ℕ²

    Uses continuous relaxation followed by rounding to nearest integer,
    with mechanisms to handle infeasible integer solutions.
    """

    xd: np.ndarray  # Discrete candidate solution in log-space

    def __init__(self, params, elasticities, price_out) -> None:
        """Initialize discrete oracle with underlying continuous oracle."""
        self.omega = ProfitOracle(params, elasticities, price_out)
        self.xd = np.array([0.0, 0.0])  # Initial discrete solution

    def assess_optim_q(
        self, xc: Arr, gamma: float, retry: bool
    ) -> Tuple[Cut, Arr, Optional[float], bool]:
        """Discrete optimization assessment with rounding mechanism.

        Workflow:
        1. First try continuous solution (retry=False)
        2. If infeasible, return feasibility cut
        3. If feasible, round to nearest integer and check optimality
        4. On retry (retry=True), check rounded solution optimality

        Returns:
            Tuple containing:
            - Cut information
            - Evaluation point (continuous or rounded)
            - Updated gamma value
            - Retry flag for integer feasibility check
        """
        if not retry:
            # First attempt with continuous solution
            if cut := self.omega.assess_feas(xc, gamma):
                return cut, xc, None, True

            # Round to nearest integer (with 0 → 1 protection)
            yd = np.round(np.exp(xc))
            yd[yd == 0] = 1.0
            self.xd = np.log(yd)

        # Check optimality of discrete solution
        (grad, beta), gamma_new = self.omega.assess_optim(self.xd, gamma)
        beta += grad.dot(self.xd - xc)  # Adjust for rounding difference
        return (grad, beta), self.xd, gamma_new, not retry
