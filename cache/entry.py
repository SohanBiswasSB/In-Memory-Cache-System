"""A cache entry: the stored value plus when it expires."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

V = TypeVar("V")

_NEVER = math.inf


@dataclass(frozen=True, slots=True)
class CacheEntry(Generic[V]):
    """
    Stores an absolute deadline rather than a duration, so expiry is one float
    comparison and does not depend on when the entry was last read.
    """

    value: V
    expires_at: float = _NEVER

    @classmethod
    def immortal(cls, value: V) -> "CacheEntry[V]":
        """An entry with no TTL. It can still be evicted for capacity."""
        return cls(value, _NEVER)

    @classmethod
    def expiring_at(cls, value: V, expires_at: float) -> "CacheEntry[V]":
        """An entry that stops being visible at the absolute time `expires_at`."""
        return cls(value, expires_at)

    def is_expired(self, now: float) -> bool:
        """True once `now` reaches the deadline. Inclusive, so a zero length TTL
        is never briefly live."""
        return now >= self.expires_at
