"""
Round-robin index helper for cyclic constraint scanning.

Extracts the repeated ``idx += 1; if idx == N: idx = 0`` idiom into a small
stateful helper (used by LowpassOracle and ProfitOracle).
"""

from __future__ import annotations


class RoundRobin:
    """Round-robin index generator over a half-open range [lo, hi).

    Successive calls to :meth:`next` yield lo, lo+1, ..., hi-1, lo, ...
    Used to scan constraints cyclically.

    Args:
        n: Upper bound (exclusive); range is [0, n).
        lo: Lower bound (inclusive). Defaults to 0.
        hi: Upper bound (exclusive). If omitted, equals ``n``.
        start: Initial cursor position; the first :meth:`next` returns
            ``start + 1`` wrapped into [lo, hi). Defaults to ``lo - 1``,
            so the first call returns ``lo``.

    Examples:
        >>> rr = RoundRobin(3)
        >>> [rr.next() for _ in range(5)]
        [0, 1, 2, 0, 1]
        >>> rr2 = RoundRobin(4, lo=1)
        >>> [rr2.next() for _ in range(4)]
        [1, 2, 3, 1]
    """

    __slots__ = ("_lo", "_hi", "_cur")

    def __init__(
        self, n: int, lo: int = 0, hi: int | None = None, start: int | None = None
    ) -> None:
        self._lo = lo
        self._hi = hi if hi is not None else n
        self._cur = start if start is not None else lo - 1

    def next(self) -> int:
        """Advance to the next index and return it."""
        self._cur += 1
        if self._cur == self._hi:
            self._cur = self._lo
        return self._cur

    def peek_next(self) -> int:
        """Return the index the next :meth:`next` would yield, without advancing.

        Examples:
            >>> rr = RoundRobin(3)
            >>> rr.peek_next()
            0
            >>> rr.next()
            0
            >>> rr.peek_next()
            1
        """
        nxt = self._cur + 1
        return self._lo if nxt == self._hi else nxt

    def seek(self, idx: int) -> None:
        """Set the cursor to ``idx`` so the next :meth:`next` yields ``idx + 1``.

        Used to resume a scan right after the point at ``idx``.

        Examples:
            >>> rr = RoundRobin(4)
            >>> rr.seek(2)
            >>> [rr.next() for _ in range(3)]
            [3, 0, 1]
        """
        self._cur = idx
