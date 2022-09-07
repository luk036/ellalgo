# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from ellalgo.cutting_plane import cutting_plane_optim, cutting_plane_q
from ellalgo.ell import ell
from ellalgo.oracles.profit_oracle import (
    profit_oracle,
    profit_q_oracle,
    profit_rb_oracle,
)

p, A, k = 20.0, 40.0, 30.5
params = p, A, k
alpha, beta = 0.1, 0.4
v1, v2 = 10.0, 35.0
a = np.array([alpha, beta])
v = np.array([v1, v2])
r = np.array([100.0, 100.0])  # initial ellipsoid (sphere)


def test_profit():
    E = ell(r, np.array([0.0, 0.0]))
    P = profit_oracle(params, a, v)
    x, _, num_iters, _ = cutting_plane_optim(P, E, 0.0)
    assert x is not None
    assert num_iters == 37


def test_profit_rb():
    e1 = 0.003
    e2 = 0.007
    e3 = e4 = e5 = 1.0
    E = ell(r, np.array([0.0, 0.0]))
    P = profit_rb_oracle(params, a, v, (e1, e2, e3, e4, e5))
    x, _, num_iters, _ = cutting_plane_optim(P, E, 0.0)
    assert x is not None
    assert num_iters == 42


def test_profit_q():
    E = ell(r, np.array([0.0, 0.0]))
    P = profit_q_oracle(params, a, v)
    x, _, num_iters, _ = cutting_plane_q(P, E, 0.0)
    assert x is not None
    assert num_iters == 28
