"""
Test Example 3
"""

from __future__ import print_function

import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, Options, bsearch
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas2


class MyOracle3(OracleFeas2):
    """
    The `MyOracle3` class defines functions and gradients for mathematical operations, with a method to
    assess feasibility based on positive function values and corresponding gradients.
    """

    idx = 0
    target = -1e100

    def fn1(self, x, _):
        """
        The function fn1 returns the negation of x minus 1.

        :param x: The parameter `x` represents a value that must be greater than or equal to -1, as per the
        constraint provided in the code snippet
        :param _: The underscore symbol "_" is commonly used as a placeholder for a variable that will not
        be used in the function. In this case, it seems that the second parameter of the function `fn1` is
        not being used in the calculation, so it is represented by "_"
        :return: The function `fn1` is returning the value `-x - 1`.
        """
        return -x - 1

    def fn2(self, _, y):
        """
        The function returns the negation of the input y minus 2.

        :param _: The underscore symbol "_" is typically used as a placeholder for a variable that is not
        going to be used in the function. In this case, it seems that the function `fn2` does not use the
        first parameter, so it is represented by "_"
        :param y: The parameter "y" represents a value that must be greater than or equal to -2 in the
        context of the function "fn2"
        :return: The function `fn2` is returning the value `-y - 2`.
        """
        return -y - 2

    def fn3(self, x, y):
        """
        The function `fn3` returns the sum of `x` and `y` minus 1.

        :param x: The parameter x is a variable that is used in the function fn3. It is an input value that
        is used in the calculation along with the parameter y
        :param y: The constraint for the parameter y is that it must be greater than or equal to -2
        :return: The function `fn3` is returning the value of `x + y - 1`.
        """
        return x + y - 1

    def fn4(self, x, y):
        """
        The function `fn4` takes two arguments `x` and `y`, and returns the result of `2 * x - 3 * y -
        self.target`.

        :param x: The parameter `x` in the function `fn4` represents the first input value that will be used
        in the calculation
        :param y: It looks like the function `fn4` takes two parameters `x` and `y`. The value of `y` is
        used in the calculation inside the function to perform the operation `2 * x - 3 * y - self.target`.
        If you have a specific value for `y`
        :return: The function `fn4` is returning the result of the expression `2 * x - 3 * y - self.target`.
        """
        return 2 * x - 3 * y - self.target

    def grad1(self):
        """
        The function `grad1` returns a NumPy array `[-1.0, 0.0]`.
        :return: The function `grad1` is returning a NumPy array `[-1.0, 0.0]`.
        """
        return np.array([-1.0, 0.0])

    def grad2(self):
        """
        The function `grad2` returns a NumPy array with values [0.0, -1.0].
        :return: The function `grad2` is returning a NumPy array with values `[0.0, -1.0]`.
        """
        return np.array([0.0, -1.0])

    def grad3(self):
        """
        The function `grad3` returns a NumPy array with two elements, both set to 1.0.
        :return: An array containing the values [1.0, 1.0] is being returned.
        """
        return np.array([1.0, 1.0])

    def grad4(self):
        """
        The function `grad4` returns a NumPy array with two elements: 2.0 and -3.0.
        :return: The `grad4` function is returning a NumPy array with values `[2.0, -3.0]`.
        """
        return np.array([2.0, -3.0])

    def __init__(self):
        """
        The function initializes a class instance with attributes for four functions and their corresponding
        gradients.
        """
        self.fns = (self.fn1, self.fn2, self.fn3, self.fn4)
        self.grads = (self.grad1, self.grad2, self.grad3, self.grad4)

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

        for _ in range(4):
            self.idx += 1
            if self.idx == 4:
                self.idx = 0  # round robin
            if (fj := self.fns[self.idx](x, y)) > 0:
                return self.grads[self.idx](), fj
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
