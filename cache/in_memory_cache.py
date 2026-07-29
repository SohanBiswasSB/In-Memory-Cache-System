"""A bounded, TTL-aware, thread-safe in-memory cache."""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import threading
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass
from typing import Any, Optional, TypeVar, cast

from .base import Cache, CacheStats, RemovalReason
from .clock import MONOTONIC_CLOCK, Clock
from .entry import CacheEntry
from .eviction import EvictionPolicy, LRUEvictionPolicy

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_LOG = logging.getLogger(__name__)

# Sentinel for "no value", so a stored None is distinguishable from a miss.
MISSING: Any = object()


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

    Reads, writes, and eviction are O(1). Entries live in a dict, recency lives
    in the eviction policy, and deadlines live in a heap so expired keys can be
    found without scanning the whole cache.

    Expiry is lazy: an entry is reclaimed when it is touched, and when the cache
    needs room. To reclaim memory held by keys that are never read again, run
    purge_expired() on a timer or use ExpiryReaper.

    Every public method takes a re-entrant lock, so the cache is one lock wide.
    Loaders and removal listeners are called with the lock released, so user code
    cannot deadlock the cache or hold up other threads.
    """

    def __init__(
        self,
        capacity: int,
        *,
        default_ttl: Optional[float] = None,
        clock: Clock = MONOTONIC_CLOCK,
        eviction_policy: Optional[EvictionPolicy[K]] = None,
        refresh_ttl_on_access: bool = False,
        max_weight: Optional[float] = None,
        weigher: Optional[Callable[[K, V], float]] = None,
        removal_listener: Optional[Callable[[K, V, RemovalReason], None]] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if default_ttl is not None:
            _validate_ttl(default_ttl)
        if max_weight is not None and max_weight <= 0:
            raise ValueError(f"max_weight must be positive, got {max_weight}")
        if weigher is not None and max_weight is None:
            raise ValueError("weigher is only meaningful together with max_weight")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._clock = clock
        # Compared against None rather than truth-tested: a policy defines
        # __len__, so an empty one is falsy and `or` would discard it.
        self._policy: EvictionPolicy[K] = (
            LRUEvictionPolicy() if eviction_policy is None else eviction_policy
        )
        self._refresh_on_access = refresh_ttl_on_access
        self._max_weight = max_weight
        self._weigher = weigher
        self._listener = removal_listener

        self._entries: dict[K, CacheEntry[V]] = {}
        self._lock = threading.RLock()
        self._counters = _Counters()
        self._weight = 0.0

        # (deadline, tiebreak, key) for every entry written with a finite TTL.
        # The counter keeps tuple comparison off the keys, which need not be
        # orderable. Records superseded by a later write are spotted when popped
        # rather than removed on write, which would cost O(n).
        self._deadlines: list[tuple[float, int, K]] = []
        self._tiebreak = itertools.count()

        # Per-key locks for get_or_load, so only one loader runs per key.
        self._loading: dict[K, tuple[threading.Lock, int]] = {}

        # Removals waiting to be reported once the lock is released.
        self._pending: list[tuple[K, V, RemovalReason]] = []

    # --- reads ---------------------------------------------------------

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            result = self._get_locked(key, default)
        self._dispatch()
        return result

    def _get_locked(self, key: K, default: Optional[V]) -> Optional[V]:
        entry = self._entries.get(key)

        if entry is None:
            self._counters.misses += 1
            return default

        now = self._clock()
        if entry.is_expired(now):
            self._discard(key, RemovalReason.EXPIRED)
            self._counters.misses += 1
            return default

        self._policy.on_access(key)  # a read counts as recent use
        if self._refresh_on_access and entry.ttl is not None:
            self._entries[key] = entry.renewed(now)
            self._track_deadline(key, self._entries[key])
            self._compact_deadlines()
        self._counters.hits += 1
        return entry.value

    def get_or_load(
        self,
        key: K,
        loader: Callable[[], V],
        ttl: Optional[float] = None,
    ) -> V:
        """
        Return the cached value, or call `loader` once and cache what it returns.

        Concurrent callers for the same key wait for the first loader instead of
        all calling it, which is what stops a hot key expiring from becoming a
        thundering herd on whatever sits behind the cache. Different keys never
        block each other, and `loader` runs without the cache lock held.
        """
        value = self.get(key, MISSING)
        if value is not MISSING:
            return cast(V, value)

        lock = self._acquire_key_lock(key)
        try:
            with lock:
                # Someone may have loaded it while we waited for this lock.
                value = self.get(key, MISSING)
                if value is not MISSING:
                    return cast(V, value)
                loaded = loader()
                self.put(key, loaded, ttl)
                return loaded
        finally:
            self._release_key_lock(key)

    def __contains__(self, key: object) -> bool:
        """Membership test that does not affect recency."""
        with self._lock:
            entry = self._entries.get(cast(K, key))
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

    @property
    def weight(self) -> float:
        """Total weight held. Equals the entry count with no weigher configured."""
        with self._lock:
            return self._weight

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
            weight = self._weigh(key, value)
            if effective_ttl is None:
                entry = CacheEntry.immortal(value, weight)
            else:
                entry = CacheEntry.expiring_at(
                    value, now + effective_ttl, effective_ttl, weight
                )

            previous = self._entries.get(key)
            if previous is not None:
                self._retire(key, previous, RemovalReason.REPLACED)

            self._entries[key] = entry
            self._weight += weight
            self._policy.on_access(key)
            self._track_deadline(key, entry)

            self._make_room(now, protected=key)
            self._compact_deadlines()
        self._dispatch()

    def remove(self, key: K) -> bool:
        with self._lock:
            present = key in self._entries
            if present:
                self._discard(key, RemovalReason.REMOVED)
        self._dispatch()
        return present

    def clear(self) -> None:
        with self._lock:
            if self._listener is not None:
                for key, entry in self._entries.items():
                    self._pending.append((key, entry.value, RemovalReason.REMOVED))
            self._entries.clear()
            self._policy.clear()
            self._deadlines.clear()
            self._weight = 0.0
        self._dispatch()

    def purge_expired(self, batch: int = 512) -> int:
        """
        Drop expired entries and return how many were reclaimed.

        Driven by the deadline heap rather than a scan of every key, and the lock
        is released between batches so a large sweep cannot stall readers for its
        whole duration.
        """
        reclaimed = 0
        while True:
            with self._lock:
                dropped = self._drain_expired(self._clock(), limit=batch)
            self._dispatch()
            reclaimed += dropped
            if dropped < batch:
                return reclaimed

    # --- internals -----------------------------------------------------

    def _weigh(self, key: K, value: V) -> float:
        if self._weigher is None:
            return 1.0
        weight = self._weigher(key, value)
        if weight <= 0:
            raise ValueError(f"weigher returned a non-positive weight: {weight!r}")
        return weight

    def _track_deadline(self, key: K, entry: CacheEntry[V]) -> None:
        """Record a finite deadline in the heap. Caller must hold the lock."""
        if math.isfinite(entry.expires_at):
            heapq.heappush(
                self._deadlines, (entry.expires_at, next(self._tiebreak), key)
            )

    def _drain_expired(self, now: float, limit: int) -> int:
        """
        Pop due deadlines and discard entries that are still expired at `now`.
        Returns how many were dropped. Caller must hold the lock.
        """
        dropped = 0
        while self._deadlines and dropped < limit:
            deadline, _, key = self._deadlines[0]
            if deadline > now:
                break
            heapq.heappop(self._deadlines)
            entry = self._entries.get(key)
            # A deadline that no longer matches its entry was superseded by a
            # later write, and that write pushed its own record.
            if entry is None or entry.expires_at != deadline:
                continue
            if entry.is_expired(now):
                self._discard(key, RemovalReason.EXPIRED)
                dropped += 1
        return dropped

    def _make_room(self, now: float, protected: Optional[K] = None) -> None:
        """
        Evict until the cache is back inside its capacity and weight limits.
        Expired entries go first because they cost nothing to lose. `protected`
        is the key just written and is never chosen. Caller must hold the lock.
        """
        while self._over_limit():
            if self._drain_expired(now, limit=1):
                continue
            victim = self._pick_victim(protected)
            if victim is None:
                return
            self._discard(victim, RemovalReason.EVICTED)
            self._counters.evictions += 1

    def _over_limit(self) -> bool:
        if len(self._entries) > self._capacity:
            return True
        return self._max_weight is not None and self._weight > self._max_weight

    def _pick_victim(self, protected: Optional[K]) -> Optional[K]:
        victim = self._policy.evict_candidate()
        if victim is None:
            return None
        if victim != protected:
            return victim
        # The coldest key is the one just written, which happens when a single
        # entry outweighs the whole limit. Take the next coldest instead.
        for candidate in self._policy.candidates():
            if candidate != protected:
                return candidate
        return None

    def _retire(self, key: K, entry: CacheEntry[V], reason: RemovalReason) -> None:
        """Account for an entry leaving while its key stays, which is what a
        write over an existing key does. Caller must hold the lock."""
        self._weight -= entry.weight
        if self._listener is not None:
            self._pending.append((key, entry.value, reason))

    def _discard(self, key: K, reason: RemovalReason) -> None:
        """Only place a key is removed, so the policy cannot drift out of sync
        with the entry dict. Caller must hold the lock."""
        try:
            entry = self._entries.pop(key)
        except KeyError:
            # The policy nominated a key this cache does not hold, so the two
            # key sets have drifted. Say so, rather than surfacing a bare
            # KeyError from deep inside an eviction.
            raise RuntimeError(
                f"eviction policy {type(self._policy).__name__} nominated {key!r}, "
                "which this cache does not hold; the policy is tracking keys from "
                "somewhere else (a policy instance must not be shared between caches)"
            ) from None
        self._policy.on_remove(key)
        self._weight -= entry.weight
        if reason is RemovalReason.EXPIRED:
            self._counters.expirations += 1
        if self._listener is not None:
            self._pending.append((key, entry.value, reason))

    def _compact_deadlines(self) -> None:
        """Rebuild the heap once superseded records outnumber live entries.
        Caller must hold the lock."""
        if len(self._deadlines) <= 2 * len(self._entries) + 32:
            return
        self._deadlines = [
            record
            for record in self._deadlines
            if (entry := self._entries.get(record[2])) is not None
            and entry.expires_at == record[0]
        ]
        heapq.heapify(self._deadlines)

    def _dispatch(self) -> None:
        """Report queued removals with the lock NOT held, so a listener may call
        back into the cache and cannot block other threads."""
        if self._listener is None:
            return
        with self._lock:
            if not self._pending:
                return
            events, self._pending = self._pending, []
        for key, value, reason in events:
            try:
                self._listener(key, value, reason)
            except Exception:  # pylint: disable=broad-exception-caught
                # A broken listener must not take the cache down with it.
                _LOG.exception("removal listener failed for key %r", key)

    def _acquire_key_lock(self, key: K) -> threading.Lock:
        with self._lock:
            lock, waiters = self._loading.get(key, (threading.Lock(), 0))
            self._loading[key] = (lock, waiters + 1)
            return lock

    def _release_key_lock(self, key: K) -> None:
        with self._lock:
            lock, waiters = self._loading[key]
            if waiters <= 1:
                del self._loading[key]
            else:
                self._loading[key] = (lock, waiters - 1)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"{type(self).__name__}(size={len(self._entries)}, "
                f"capacity={self._capacity}, policy={type(self._policy).__name__})"
            )


def _validate_ttl(ttl: float) -> None:
    if ttl <= 0:
        raise ValueError(f"ttl must be positive seconds, got {ttl!r}")
