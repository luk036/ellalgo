"""
Test Example 3
"""

from __future__ import print_function

import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, Options, bsearch
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas2

num_constraints = 4


class MyOracle3(OracleFeas2):
    """
    The `MyOracle3` class defines functions and gradients for mathematical operations, with a method to
    assess feasibility based on positive function values and corresponding gradients.
    """

    idx = -1
    target = -1e100

    def assess_feas(self, xc):
        """
        The `assess_feas` function iterates through a list of functions and returns the result of the first
        function that returns a positive value along with its corresponding gradient.

        :param xc: The `xc` parameter in the `assess_feas` method is a tuple containing two values, `x` and
            `y`. These values are then unpacked from the tuple using the line `x, y = xc` within the method
        :return: If the condition `(fj := self.fns[self.idx](x, y)) > 0` is met for any of the iterations in
            the for loop, then a tuple containing the result of `self.grads[self.idx]()` and the value of `fj`
            will be returned. Otherwise, if the condition is never met, `None` will be returned.
        """
        x, y = xc

        for _ in range(num_constraints):
            self.idx += 1
            if self.idx == num_constraints:
                self.idx = 0  # round robin

            if self.idx == 0:
                if (fj := -x - 1) > 0.0:
                    return np.array([-1.0, 0.0]), fj
            elif self.idx == 1:
                if (fj := -y - 2) > 0.0:
                    return np.array([0.0, -1.0]), fj
            elif self.idx == 2:
                if (fj := x + y - 1) > 0.0:
                    return np.array([1.0, 1.0]), fj
            elif self.idx == 3:
                if (fj := 2 * x - 3 * y - self.target) > 0.0:
                    return np.array([2.0, -3.0]), fj
            else:
                raise ValueError("Unexpected index value")

        return None

    def update(self, gamma):
        """
        The `update` function sets the `target` attribute to the value of the `gamma` parameter.

        :param gamma: Gamma is a parameter used in reinforcement learning algorithms, specifically in the
            context of discounted rewards. It represents the discount factor, which determines the importance of
            future rewards in relation to immediate rewards. A gamma value closer to 1 gives more weight to
            future rewards, while a gamma value closer to 0 gives
        """
        self.target = gamma


def test_case_feasible():
    """
    The function `test_case_feasible` sets up a test case for binary search with specific parameters and
    asserts the expected outcome.
    """
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(100.0, xinit)
    options = Options()
    options.tolerance = 1e-8
    adaptor = BSearchAdaptor(MyOracle3(), ellip, options)
    xbest, num_iters = bsearch(adaptor, (-100.0, 100.0), options)
    assert xbest is not None
    assert num_iters == 34
