"""Cache interface, the stats record it returns, and removal reasons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class RemovalReason(Enum):
    """Why an entry left the cache. Passed to a removal listener."""

    EXPIRED = "expired"
    EVICTED = "evicted"
    REMOVED = "removed"
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Counter snapshot returned by Cache.stats()."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache; 0.0 if never queried."""
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0

    def __add__(self, other: "CacheStats") -> "CacheStats":
        """Combine two snapshots, so a sharded cache can total its shards."""
        if not isinstance(other, CacheStats):
            return NotImplemented
        return CacheStats(
            hits=self.hits + other.hits,
            misses=self.misses + other.misses,
            evictions=self.evictions + other.evictions,
            expirations=self.expirations + other.expirations,
        )


class Cache(ABC, Generic[K, V]):
    """
    A bounded key-value store with optional per-key time-to-live.

    Implementations must document their thread-safety guarantees.
    """

    @abstractmethod
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """The live value for `key`, or `default` if absent or expired."""

    @abstractmethod
    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Store `value` under `key`.

        :param ttl: seconds until the entry stops being visible. `None` means
            "use the cache default", which is itself usually no expiry. Must be
            positive; `math.inf` means never expire.
        """

    @abstractmethod
    def remove(self, key: K) -> bool:
        """True if a mapping was present and removed."""

    @abstractmethod
    def clear(self) -> None:
        """Drop every entry, resetting the cache to empty."""

    @abstractmethod
    def stats(self) -> CacheStats:
        """Snapshot of the counters."""

    @abstractmethod
    def purge_expired(self) -> int:
        """Reclaim expired entries eagerly. Returns how many were dropped."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __contains__(self, key: object) -> bool: ...
