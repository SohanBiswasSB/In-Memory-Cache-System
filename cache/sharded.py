"""A cache split into independent shards to cut lock contention."""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from typing import Any, Optional, TypeVar

from .base import Cache, CacheStats
from .eviction import EvictionPolicy
from .in_memory_cache import InMemoryCache

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class ShardedCache(Cache[K, V]):
    """
    Several InMemoryCache instances, each with its own lock, chosen by key hash.

    A single cache serialises every operation on one lock. Sharding gives up
    global LRU ordering, since each shard evicts only from its own keys, in
    exchange for roughly N-way concurrency. That approximation is usually a fair
    trade, but it does mean a shard can evict a key that is hotter than one
    another shard keeps.

    Implements the same Cache interface as InMemoryCache, so callers do not need
    to know which one they hold.
    """

    def __init__(
        self,
        capacity: int,
        *,
        shards: int = 8,
        policy_factory: Optional[Callable[[], EvictionPolicy[K]]] = None,
        **cache_options: Any,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if shards <= 0:
            raise ValueError(f"shards must be positive, got {shards}")
        if "eviction_policy" in cache_options:
            # Sharing one policy across shards is silently wrong: it would track
            # every key, so a shard could be told to evict a key it does not
            # hold. Each shard needs its own instance, hence a factory.
            raise ValueError(
                "pass policy_factory to ShardedCache rather than eviction_policy, "
                "so each shard gets its own policy instance"
            )

        per_shard = max(1, math.ceil(capacity / shards))
        self._shards: tuple[InMemoryCache[K, V], ...] = tuple(
            InMemoryCache(
                per_shard,
                **(
                    cache_options
                    if policy_factory is None
                    else {**cache_options, "eviction_policy": policy_factory()}
                ),
            )
            for _ in range(shards)
        )

    def _shard_for(self, key: K) -> InMemoryCache[K, V]:
        return self._shards[hash(key) % len(self._shards)]

    @property
    def shard_count(self) -> int:
        """How many independent caches back this one."""
        return len(self._shards)

    @property
    def capacity(self) -> int:
        """Total capacity across shards, which rounding may push above the
        requested figure."""
        return sum(shard.capacity for shard in self._shards)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        return self._shard_for(key).get(key, default)

    def get_or_load(
        self,
        key: K,
        loader: Callable[[], V],
        ttl: Optional[float] = None,
    ) -> V:
        """Load-through for one key, guarded within its own shard."""
        return self._shard_for(key).get_or_load(key, loader, ttl)

    def put(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        self._shard_for(key).put(key, value, ttl)

    def remove(self, key: K) -> bool:
        return self._shard_for(key).remove(key)

    def clear(self) -> None:
        for shard in self._shards:
            shard.clear()

    def purge_expired(self) -> int:
        return sum(shard.purge_expired() for shard in self._shards)

    def stats(self) -> CacheStats:
        """Shard totals. Taken shard by shard, so it is not a single instant."""
        total = CacheStats()
        for shard in self._shards:
            total = total + shard.stats()
        return total

    def __len__(self) -> int:
        return sum(len(shard) for shard in self._shards)

    def __contains__(self, key: object) -> bool:
        return key in self._shard_for(key)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(shards={len(self._shards)}, "
            f"capacity={self.capacity}, size={len(self)})"
        )
