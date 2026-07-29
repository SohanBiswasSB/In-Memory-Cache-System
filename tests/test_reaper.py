"""ExpiryReaper lifecycle and behaviour."""

# pylint: disable=missing-class-docstring, missing-function-docstring

from __future__ import annotations

import unittest
from typing import Optional

from cache import Cache, CacheStats, ExpiryReaper, InMemoryCache, ShardedCache

from .support import wait_until


class _CountingCache(Cache[str, int]):
    """Minimal cache that records sweeps and fails every one of them."""

    def __init__(self) -> None:
        self.sweeps = 0

    def get(self, key: str, default: Optional[int] = None) -> Optional[int]:
        return default

    def put(self, key: str, value: int, ttl: Optional[float] = None) -> None:
        pass

    def remove(self, key: str) -> bool:
        return False

    def clear(self) -> None:
        pass

    def stats(self) -> CacheStats:
        return CacheStats()

    def purge_expired(self) -> int:
        self.sweeps += 1
        raise RuntimeError("sweep is broken")

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: object) -> bool:
        return False


class TestReaperLifecycle(unittest.TestCase):
    def test_context_manager_starts_and_stops_the_thread(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4)
        reaper = ExpiryReaper(cache, interval_seconds=0.01)

        self.assertFalse(reaper.running)
        with reaper:
            self.assertTrue(reaper.running)
        self.assertFalse(reaper.running)

    def test_starting_twice_is_an_error(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4)
        reaper = ExpiryReaper(cache, interval_seconds=0.01)
        reaper.start()
        try:
            with self.assertRaises(RuntimeError):
                reaper.start()
        finally:
            reaper.stop()

    def test_stop_is_safe_to_call_twice(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4)
        reaper = ExpiryReaper(cache, interval_seconds=0.01)
        reaper.start()
        reaper.stop()
        reaper.stop()

        self.assertFalse(reaper.running)

    def test_can_be_restarted_after_stopping(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4)
        reaper = ExpiryReaper(cache, interval_seconds=0.01)
        reaper.start()
        reaper.stop()
        reaper.start()
        try:
            self.assertTrue(reaper.running)
        finally:
            reaper.stop()

    def test_rejects_a_non_positive_interval(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4)
        with self.assertRaises(ValueError):
            ExpiryReaper(cache, interval_seconds=0)


class TestReaperSweeping(unittest.TestCase):
    def test_reclaims_entries_nothing_reads_back(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=100)
        for i in range(5):
            cache.put(f"tmp:{i}", i, ttl=0.02)

        self.assertEqual(5, cache.entry_count)
        with ExpiryReaper(cache, interval_seconds=0.01):
            swept = wait_until(lambda: cache.entry_count == 0)

        self.assertTrue(swept, "reaper should have reclaimed every expired entry")

    def test_leaves_live_entries_alone(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=100)
        cache.put("keep", 1)
        cache.put("drop", 2, ttl=0.02)

        with ExpiryReaper(cache, interval_seconds=0.01):
            wait_until(lambda: "drop" not in cache)

        self.assertEqual(1, cache.get("keep"))

    def test_survives_a_failing_sweep(self) -> None:
        cache = _CountingCache()
        with self.assertLogs("cache.reaper", level="ERROR"):
            with ExpiryReaper(cache, interval_seconds=0.01):
                kept_going = wait_until(lambda: cache.sweeps >= 3)

        self.assertTrue(kept_going, "one bad sweep must not kill the thread")

    def test_works_with_a_sharded_cache(self) -> None:
        clock_free: ShardedCache[int, int] = ShardedCache(capacity=64, shards=4)
        for i in range(10):
            clock_free.put(i, i, ttl=0.02)

        with ExpiryReaper(clock_free, interval_seconds=0.01):
            swept = wait_until(lambda: len(clock_free) == 0)

        self.assertTrue(swept)


if __name__ == "__main__":
    unittest.main(verbosity=2)
