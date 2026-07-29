"""A bounded, TTL-aware, thread-safe in-memory cache."""

from __future__ import annotations

import threading
from collections.abc import Hashable
from dataclasses import asdict, dataclass
from typing import Optional, TypeVar

from .base import Cache, CacheStats
from .clock import MONOTONIC_CLOCK, Clock
from .entry import CacheEntry
from .eviction import EvictionPolicy, LRUEvictionPolicy

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(slots=True)
class _Counters:
    """Mutable counters, copied into an immutable CacheStats by stats()."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0


class InMemoryCache(Cache[K, V]):
    """
    Fixed-capacity key-value store with per-key TTL and LRU eviction.

    All operations are O(1): a dict holds the entries, and the eviction policy
    maintains recency in a linked list.

    Expiry is lazy: an entry is reclaimed when it is touched, and
    opportunistically when the cache needs room. To reclaim memory held by keys
    that are never read again, run purge_expired() on a timer or use
    ExpiryReaper.

    Every public method takes a re-entrant lock, so the cache is one lock wide.
    """

    # How many keys to check from the cold end for an already expired entry
    # before falling back to plain LRU. Bounded so eviction stays O(1).
    _EXPIRY_SCAN_LIMIT = 8

    def __init__(
        self,
        capacity: int,
        *,
        default_ttl: Optional[float] = None,
        clock: Clock = MONOTONIC_CLOCK,
        eviction_policy: Optional[EvictionPolicy[K]] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if default_ttl is not None:
            _validate_ttl(default_ttl)

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._clock = clock
        self._policy: EvictionPolicy[K] = eviction_policy or LRUEvictionPolicy()

        self._entries: dict[K, CacheEntry[V]] = {}
        self._lock = threading.RLock()
        self._counters = _Counters()

    # --- reads ---------------------------------------------------------

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._counters.misses += 1
                return default

            if entry.is_expired(self._clock()):
                self._discard(key, expired=True)
                self._counters.misses += 1
                return default

            self._policy.on_access(key)  # a read counts as recent use
            self._counters.hits += 1
            return entry.value

    def __contains__(self, key: object) -> bool:
        """Membership test that does not affect recency."""
        with self._lock:
            entry = self._entries.get(key)  # type: ignore[arg-type]
            return entry is not None and not entry.is_expired(self._clock())

    def __len__(self) -> int:
        """Live entries only; expired-but-unreclaimed keys are not counted."""
        with self._lock:
            now = self._clock()
            return sum(1 for e in self._entries.values() if not e.is_expired(now))

    @property
    def capacity(self) -> int:
        """Maximum number of entries held before eviction kicks in."""
        return self._capacity

    @property
    def entry_count(self) -> int:
        """Slots occupied, including expired entries not yet reclaimed. Use this
        for memory; use len() for what callers can actually read."""
        with self._lock:
            return len(self._entries)

    def stats(self) -> CacheStats:
        """Immutable snapshot of the counters, read atomically under the lock."""
        with self._lock:
            return CacheStats(**asdict(self._counters))

    # --- writes --------------------------------------------------------

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        `ttl=None` falls back to the cache-wide default; pass ``math.inf`` to
        pin an entry against expiry when a default TTL is configured.
        """
        effective_ttl = self._default_ttl if ttl is None else ttl
        if effective_ttl is not None:
            _validate_ttl(effective_ttl)

        with self._lock:
            now = self._clock()
            entry = (
                CacheEntry.immortal(value)
                if effective_ttl is None
                else CacheEntry.expiring_at(value, now + effective_ttl)
            )

            if key not in self._entries and len(self._entries) >= self._capacity:
                self._make_room(now)

            self._entries[key] = entry  # overwrite refreshes value and TTL
            self._policy.on_access(key)

    def remove(self, key: K) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._discard(key)
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._policy.clear()

    def purge_expired(self) -> int:
        """
        Drop every expired entry and return how many were reclaimed. O(n), so
        this is for a background sweep rather than the hot path.
        """
        with self._lock:
            now = self._clock()
            dead = [k for k, e in self._entries.items() if e.is_expired(now)]
            for key in dead:
                self._discard(key, expired=True)
            return len(dead)

    # --- internals -----------------------------------------------------

    def _make_room(self, now: float) -> None:
        """
        Free exactly one slot. An already expired key costs nothing to lose, so
        prefer one of those and only fall back to the true LRU. Caller must hold
        the lock.
        """
        for scanned, key in enumerate(self._policy.candidates()):
            if self._entries[key].is_expired(now):
                self._discard(key, expired=True)
                return
            if scanned + 1 >= self._EXPIRY_SCAN_LIMIT:
                break

        victim = self._policy.evict_candidate()
        if victim is not None:
            self._discard(victim, evicted=True)

    def _discard(self, key: K, *, expired: bool = False, evicted: bool = False) -> None:
        """Only place a key is removed, so the policy cannot drift out of sync
        with the entry dict. Caller must hold the lock."""
        del self._entries[key]
        self._policy.on_remove(key)
        if expired:
            self._counters.expirations += 1
        if evicted:
            self._counters.evictions += 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        with self._lock:
            return (
                f"{type(self).__name__}(size={len(self._entries)}, "
                f"capacity={self._capacity}, policy={type(self._policy).__name__})"
            )


def _validate_ttl(ttl: float) -> None:
    if ttl <= 0:
        raise ValueError(f"ttl must be positive seconds, got {ttl!r}")
