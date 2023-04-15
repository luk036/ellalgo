# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim, cutting_plane_q
from ellalgo.ell import Ell
from ellalgo.oracles.profit_oracle import ProfitOracle, ProfitQOracle, ProfitRbOracle

p, A, k = 20.0, 40.0, 30.5
params = p, A, k
alpha, beta = 0.1, 0.4
v1, v2 = 10.0, 35.0
a = np.array([alpha, beta])
v = np.array([v1, v2])
r = np.array([100.0, 100.0])  # initial ellipsoid (sphere)


def test_profit():
    ellip = Ell(r, np.array([0.0, 0.0]))
    omega = ProfitOracle(params, a, v)
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    assert num_iters == 36


def test_profit_rb():
    e1 = 0.003
    e2 = 0.007
    e3 = e4 = e5 = 1.0
    ellip = Ell(r, np.array([0.0, 0.0]))
    omega = ProfitRbOracle(params, a, v, (e1, e2, e3, e4, e5))
    xbest, _, num_iters = cutting_plane_optim(omega, ellip, 0.0)
    assert xbest is not None
    assert num_iters == 41


def test_profit_q():
    ellip = Ell(r, np.array([0.0, 0.0]))
    omega = ProfitQOracle(params, a, v)
    xbest, _, num_iters = cutting_plane_q(omega, ellip, 0.0)
    assert xbest is not None
    assert num_iters == 27
