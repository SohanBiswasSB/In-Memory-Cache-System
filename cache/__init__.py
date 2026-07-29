"""
In-memory key/value cache with per-key TTL and LRU eviction.

    from cache import InMemoryCache

    c = InMemoryCache(capacity=1000)
    c.put("session:42", {"user": "sohan"}, ttl=30)
    c.get("session:42")
"""

from .base import Cache, CacheStats
from .clock import MONOTONIC_CLOCK, Clock, ManualClock
from .entry import CacheEntry
from .eviction import EvictionPolicy, LRUEvictionPolicy
from .in_memory_cache import InMemoryCache
from .reaper import ExpiryReaper

__all__ = [
    "Cache",
    "CacheEntry",
    "CacheStats",
    "Clock",
    "EvictionPolicy",
    "ExpiryReaper",
    "InMemoryCache",
    "LRUEvictionPolicy",
    "MONOTONIC_CLOCK",
    "ManualClock",
]
