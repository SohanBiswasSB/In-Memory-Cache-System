"""
In-memory key/value cache with per-key TTL and LRU eviction.

    from cache import InMemoryCache

    c = InMemoryCache(capacity=1000)
    c.put("session:42", {"user": "sohan"}, ttl=30)
    c.get("session:42")
"""

from .base import Cache, CacheStats, RemovalReason
from .clock import MONOTONIC_CLOCK, Clock, ManualClock
from .entry import CacheEntry
from .eviction import EvictionPolicy, LFUEvictionPolicy, LRUEvictionPolicy
from .in_memory_cache import InMemoryCache
from .reaper import ExpiryReaper
from .sharded import ShardedCache

__all__ = [
    "Cache",
    "CacheEntry",
    "CacheStats",
    "Clock",
    "EvictionPolicy",
    "ExpiryReaper",
    "InMemoryCache",
    "LFUEvictionPolicy",
    "LRUEvictionPolicy",
    "MONOTONIC_CLOCK",
    "ManualClock",
    "RemovalReason",
    "ShardedCache",
]
