"""
Rough throughput numbers: python benchmark.py

The last section is the one that matters: if lookups really are O(1), the rate
should stay flat as the cache grows by three orders of magnitude.
"""

from __future__ import annotations

import random
import time
from typing import Callable

from cache import InMemoryCache


def measure(label: str, operation: Callable[[int], object], iterations: int) -> float:
    """Run `operation` and report the rate. Returns operations per second."""
    start = time.perf_counter()
    for i in range(iterations):
        operation(i)
    elapsed = time.perf_counter() - start
    rate = iterations / elapsed
    print(f"  {label:<38}{rate:>14,.0f} ops/sec")
    return rate


def bench_writes() -> None:
    """Write throughput, with and without eviction on every call."""
    print("writes")
    roomy: InMemoryCache[int, int] = InMemoryCache(capacity=2_000_000)
    measure("put, never evicting", lambda i: roomy.put(i, i), 200_000)

    tight: InMemoryCache[int, int] = InMemoryCache(capacity=1_000)
    measure("put, evicting every time", lambda i: tight.put(i, i), 200_000)

    with_ttl: InMemoryCache[int, int] = InMemoryCache(capacity=2_000_000)
    measure("put with a TTL", lambda i: with_ttl.put(i, i, ttl=3600), 200_000)


def bench_reads() -> None:
    """Read throughput for hits, misses, and the load-through path."""
    print("reads")
    cache: InMemoryCache[int, int] = InMemoryCache(capacity=200_000)
    for i in range(100_000):
        cache.put(i, i)

    measure("get, hit", lambda i: cache.get(i % 100_000), 200_000)
    measure("get, miss", lambda i: cache.get(-i - 1), 200_000)
    measure("contains", lambda i: (i % 100_000) in cache, 200_000)
    measure(
        "get_or_load, hit",
        lambda i: cache.get_or_load(i % 100_000, lambda: 0),
        200_000,
    )

    sliding: InMemoryCache[int, int] = InMemoryCache(
        capacity=200_000, refresh_ttl_on_access=True
    )
    for i in range(100_000):
        sliding.put(i, i, ttl=3600)
    measure("get, hit, sliding TTL", lambda i: sliding.get(i % 100_000), 200_000)


def bench_expiry() -> None:
    """How fast a sweep reclaims a cache full of dead entries."""
    print("expiry")
    cache: InMemoryCache[int, int] = InMemoryCache(capacity=500_000)
    for i in range(200_000):
        cache.put(i, i, ttl=0.001)
    time.sleep(0.01)

    start = time.perf_counter()
    reclaimed = cache.purge_expired()
    elapsed = time.perf_counter() - start
    print(f"  {'purge_expired':<38}{reclaimed / elapsed:>14,.0f} entries/sec")


def bench_scaling() -> None:
    """The claim that lookups are O(1), measured rather than asserted."""
    print("lookup rate as the cache grows (should stay flat)")
    for size in (1_000, 10_000, 100_000, 1_000_000):
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=size * 2)
        for i in range(size):
            cache.put(i, i)
        keys = [random.randrange(size) for _ in range(50_000)]
        start = time.perf_counter()
        for key in keys:
            cache.get(key)
        elapsed = time.perf_counter() - start
        print(f"  {size:>9,} entries{len(keys) / elapsed:>26,.0f} ops/sec")


if __name__ == "__main__":
    random.seed(0)
    bench_writes()
    bench_reads()
    bench_expiry()
    bench_scaling()
