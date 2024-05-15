# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from ellalgo.cutting_plane import BSearchAdaptor, Options, bsearch
from ellalgo.ell import Ell
from ellalgo.ell_typing import OracleFeas2


class MyOracle3(OracleFeas2):
    idx = 0
    target = -1e100

    # constraint 1: x >= -1
    def fn1(self, x, _):
        return -x - 1

    # constraint 2: y >= -2
    def fn2(self, _, y):
        return -y - 2

    # constraint 2: y >= -2
    def fn3(self, x, y):
        return x + y - 1

    def fn4(self, x, y):
        return 2 * x - 3 * y - self.target

    def grad1(self):
        return np.array([-1.0, 0.0])

    def grad2(self):
        return np.array([0.0, -1.0])

    def grad3(self):
        return np.array([1.0, 1.0])

    def grad4(self):
        return np.array([2.0, -3.0])

    def __init__(self):
        self.fns = (self.fn1, self.fn2, self.fn3, self.fn4)
        self.grads = (self.grad1, self.grad2, self.grad3, self.grad4)

    def assess_feas(self, xc):
        x, y = xc

        for _ in range(4):
            self.idx += 1
            if self.idx == 4:
                self.idx = 0  # round robin
            if (fj := self.fns[self.idx](x, y)) > 0:
                return self.grads[self.idx](), fj
        return None

    def update(self, gamma):
        self.target = gamma


def test_case_feasible():
    xinit = np.array([0.0, 0.0])  # initial xinit
    ellip = Ell(100.0, xinit)
    options = Options()
    options.tolerance = 1e-8
    adaptor = BSearchAdaptor(MyOracle3(), ellip, options)
    xbest, num_iters = bsearch(adaptor, (-100.0, 100.0), options)
    assert xbest is not None
    assert num_iters == 34
