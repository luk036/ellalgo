# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim, cutting_plane_q
from ellalgo.ell import Ell
from ellalgo.oracles.profit_oracle import (
    ProfitOracle,
    ProfitQOracle,
    ProfitRbOracle,
)

p, A, k = 20.0, 40.0, 30.5
params = p, A, k
alpha, beta = 0.1, 0.4
v1, v2 = 10.0, 35.0
a = np.array([alpha, beta])
v = np.array([v1, v2])
r = np.array([100.0, 100.0])  # initial ellipsoid (sphere)


def test_profit():
    E = Ell(r, np.array([0.0, 0.0]))
    P = ProfitOracle(params, a, v)
    x, _, num_iters, _ = cutting_plane_optim(P, E, 0.0)
    assert x is not None
    assert num_iters == 36


def test_profit_rb():
    e1 = 0.003
    e2 = 0.007
    e3 = e4 = e5 = 1.0
    E = Ell(r, np.array([0.0, 0.0]))
    P = ProfitRbOracle(params, a, v, (e1, e2, e3, e4, e5))
    x, _, num_iters, _ = cutting_plane_optim(P, E, 0.0)
    assert x is not None
    assert num_iters == 41


def test_profit_q():
    E = Ell(r, np.array([0.0, 0.0]))
    P = ProfitQOracle(params, a, v)
    x, _, num_iters, _ = cutting_plane_q(P, E, 0.0)
    assert x is not None
    assert num_iters == 27
