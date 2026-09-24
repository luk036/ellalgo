from typing import Tuple

from .ell_config import CutStatus


# The `ell1d` class represents a one-dimensional ellipse with attributes for the radius, center, and
# total squared distance.
class ell1d:
    __slots__ = ("_rd", "_xc", "_tsq")

    def __init__(self, interval: Tuple[float, float]) -> None:
        """
        The function initializes the instance variables `_rd`, `_xc`, and `_tsq` based on the given
        interval.

        :param interval: The `interval` parameter is a tuple of two floats representing the lower and
            upper bounds of a range

        :type interval: Tuple[float, float]
        """
        l, u = interval
        self._rd: float = (u - l) / 2
        self._xc: float = l + self._rd
        self._tsq: float = 0

    # def copy(self):
    #     """[summary]

    #     Returns:
    #         [type]: [description]
    #     """
    #     ellip = ell1d([self._xc - self._rd, self._xc + self._rd])
    #     return ellip

    # @property
    def xc(self) -> float:
        """
        The function `xc` returns the value of the private attribute `_xc`.
        :return: the value of the variable `self._xc`, which is of type float.
        """
        return self._xc

    # @xc.setter
    def set_xc(self, x: float) -> None:
        """
        The function sets the value of the private variable `_xc` to the given float value `x`.

        :param x: The parameter `x` is a float value that represents the value to be assigned to the
            `_xc` attribute
        :type x: float
        """
        self._xc = x

    # @property
    def tsq(self) -> float:
        """
        The function `tsq` returns the measure of the distance between `xc` and `x*`.
        :return: The method is returning a float value, which represents the measure of the distance between xc and x*.
        """
        return self._tsq

    def update(self, cut: Tuple[float, float], central_cut=False) -> CutStatus:
        """
        The `update` function updates an ellipsoid core using a single cut.

        :param cut: The `cut` parameter is a tuple containing two floats: `grad` and `beta`. `grad`
            represents the gradient, and `beta` represents the beta value

        :type cut: Tuple[float, float]

        :param central_cut: A boolean parameter that indicates whether the cut is a central cut or not.
            If it is set to True, the cut is a central cut. If it is set to False, the cut is not a central
            cut, defaults to False (optional)

        :return: a `CutStatus` enum value and the "volumn" of the ellipsoid (`tau`).

        Examples:
            >>> ellip = ell1d([0, 1])
            >>> ellip.update((1, 0))
            <CutStatus.Success: 0>
            >>> ellip.update((-1, 0))
            <CutStatus.Success: 0>
            >>> ellip.update((0, 1))
            <CutStatus.NoSoln: 1>
            >>> ellip.update((0, -1))
            <CutStatus.NoEffect: 2>
        """
        grad, beta = cut
        # TODO handle grad == 0
        tau = abs(self._rd * grad)
        self._tsq = tau**2
        # TODO: Support parallel cut
        if central_cut or beta == 0:
            self._rd /= 2
            self._xc += -self._rd if grad > 0 else self._rd
            return CutStatus.Success
        if beta > tau:
            return CutStatus.NoSoln  # no sol'n
        if beta < -tau:  # unlikely
            return CutStatus.NoEffect  # no effect

        bound = self._xc - beta / grad
        upper = bound if grad > 0 else self._xc + self._rd
        lower = self._xc - self._rd if grad > 0 else bound
        self._rd = (upper - lower) / 2
        self._xc = lower + self._rd
        return CutStatus.Success
