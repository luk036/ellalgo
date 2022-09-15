# -*- coding: utf-8 -*-
import math
from typing import Tuple, Union

import numpy as np

from .cutting_plane import CutStatus

Arr = Union[np.ndarray]

class ell1d:
    __slots__ = ("_r", "_xc")

    def __init__(self, Interval):
        """[summary]

        Arguments:
            I ([type]): [description]
        """
        l, u = Interval
        self._r = (u - l) / 2
        self._xc = l + self._r

    def copy(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        E = ell1d([self._xc - self._r, self._xc + self._r])
        return E

    @property
    def xc(self):
        """[summary]

        Returns:
            float: [description]
        """
        return self._xc

    @xc.setter
    def xc(self, x):
        """[summary]

        Arguments:
            x (float): [description]
        """
        self._xc = x

    def update(self, cut):
        """Update ellipsoid core function using the cut
                g' * (x - xc) + beta <= 0

        Arguments:
            g (floay): cut
            beta (array or scalar): [description]

        Returns:
            status: 0: success
            tau: "volumn" of ellipsoid
        """
        g, beta = cut
        # TODO handle g == 0
        tau = abs(self._r * g)
        tsq = tau**2
        if beta == 0:
            self._r /= 2
            self._xc += -self._r if g > 0 else self._r
            return CutStatus.Success, tsq
        if beta > tau:
            return CutStatus.NoSoln, tsq  # no sol'n
        if beta < -tau:  # unlikely
            return CutStatus.NoEffect, tsq  # no effect

        bound = self._xc - beta / g
        upper = bound if g > 0 else self._xc + self._r
        lower = self._xc - self._r if g > 0 else bound
        self._r = (upper - lower) / 2
        self._xc = lower + self._r
        return CutStatus.Success, tsq
