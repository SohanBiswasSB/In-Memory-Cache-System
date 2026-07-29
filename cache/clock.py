"""Time sources. Injected so TTL behaviour can be tested without sleeping."""

from __future__ import annotations

import time
from typing import Final, Protocol


class Clock(Protocol):
    """Returns a monotonically increasing number of seconds."""

    def __call__(self) -> float: ...


# monotonic rather than time.time: wall clock time can jump backwards on an NTP
# correction or a DST change, which would silently extend every live TTL.
MONOTONIC_CLOCK: Final[Clock] = time.monotonic


class ManualClock:
    """Clock for tests. Advances only when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Move virtual time forward."""
        if seconds < 0:
            raise ValueError("cannot move a monotonic clock backwards")
        self._now += seconds
