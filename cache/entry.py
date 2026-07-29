"""A cache entry: the stored value plus when it expires."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

V = TypeVar("V")

_NEVER = math.inf


@dataclass(frozen=True, slots=True)
class CacheEntry(Generic[V]):
    """
    Stores an absolute deadline rather than a duration, so expiry is one float
    comparison and does not depend on when the entry was last read.

    `ttl` keeps the original lifetime so a sliding-expiry cache can push the
    deadline out again on access. `weight` is what a weight-limited cache counts
    instead of entries.
    """

    value: V
    expires_at: float = _NEVER
    ttl: Optional[float] = None
    weight: float = 1.0

    @classmethod
    def immortal(cls, value: V, weight: float = 1.0) -> "CacheEntry[V]":
        """An entry with no TTL. It can still be evicted for capacity."""
        return cls(value, _NEVER, None, weight)

    @classmethod
    def expiring_at(
        cls,
        value: V,
        expires_at: float,
        ttl: Optional[float] = None,
        weight: float = 1.0,
    ) -> "CacheEntry[V]":
        """An entry that stops being visible at the absolute time `expires_at`."""
        return cls(value, expires_at, ttl, weight)

    def is_expired(self, now: float) -> bool:
        """True once `now` reaches the deadline. Inclusive, so a zero length TTL
        is never briefly live."""
        return now >= self.expires_at

    def renewed(self, now: float) -> "CacheEntry[V]":
        """Same value with the deadline pushed out by the original TTL. Returns
        self when there is nothing to renew."""
        if self.ttl is None:
            return self
        return CacheEntry(self.value, now + self.ttl, self.ttl, self.weight)
