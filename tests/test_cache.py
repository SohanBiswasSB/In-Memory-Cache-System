"""Tests: pytest, or python -m unittest discover -s tests -t .

Docstring checks are off because the method names say what they check. The
import position check is off because the path setup has to run first.
"""

# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=wrong-import-position

from __future__ import annotations

import math
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cache import InMemoryCache, ManualClock  # noqa: E402


class TestBasicOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.cache: InMemoryCache[str, int] = InMemoryCache(capacity=3)

    def test_put_then_get_returns_value(self) -> None:
        self.cache.put("a", 1)
        self.assertEqual(1, self.cache.get("a"))

    def test_get_missing_key_returns_default(self) -> None:
        self.assertIsNone(self.cache.get("nope"))
        self.assertEqual(-1, self.cache.get("nope", default=-1))

    def test_put_overwrites_existing_key_without_growing(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("a", 2)
        self.assertEqual(2, self.cache.get("a"))
        self.assertEqual(1, len(self.cache))

    def test_remove(self) -> None:
        self.cache.put("a", 1)
        self.assertTrue(self.cache.remove("a"))
        self.assertFalse(self.cache.remove("a"))
        self.assertNotIn("a", self.cache)

    def test_clear(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.clear()
        self.assertEqual(0, len(self.cache))

    def test_capacity_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryCache(capacity=0)

    def test_ttl_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self.cache.put("a", 1, ttl=0)
        with self.assertRaises(ValueError):
            self.cache.put("a", 1, ttl=-5)


class TestLruEviction(unittest.TestCase):
    def setUp(self) -> None:
        self.cache: InMemoryCache[str, int] = InMemoryCache(capacity=2)

    def test_evicts_least_recently_used_on_overflow(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("c", 3)  # "a" is the oldest touch -> out

        self.assertNotIn("a", self.cache)
        self.assertEqual(2, self.cache.get("b"))
        self.assertEqual(3, self.cache.get("c"))
        self.assertEqual(1, self.cache.stats().evictions)

    def test_read_refreshes_recency(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.get("a")  # "a" is now the most recent, "b" the oldest
        self.cache.put("c", 3)

        self.assertIn("a", self.cache)
        self.assertNotIn("b", self.cache)

    def test_overwrite_refreshes_recency(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("a", 11)
        self.cache.put("c", 3)

        self.assertEqual(11, self.cache.get("a"))
        self.assertNotIn("b", self.cache)

    def test_membership_check_does_not_affect_recency(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.assertIn("a", self.cache)  # must not promote "a"
        self.cache.put("c", 3)

        self.assertNotIn("a", self.cache)

    def test_never_exceeds_capacity(self) -> None:
        for i in range(100):
            self.cache.put(f"k{i}", i)
        self.assertEqual(2, len(self.cache))


class TestTimeToLive(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = ManualClock()
        self.cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=10, clock=self.clock
        )

    def test_value_visible_before_ttl_elapses(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(9.9)
        self.assertEqual(1, self.cache.get("a"))

    def test_value_gone_once_ttl_elapses(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.assertIsNone(self.cache.get("a"))
        self.assertNotIn("a", self.cache)
        self.assertEqual(1, self.cache.stats().expirations)

    def test_entry_without_ttl_never_expires(self) -> None:
        self.cache.put("a", 1)
        self.clock.advance(10_000)
        self.assertEqual(1, self.cache.get("a"))

    def test_reading_does_not_extend_ttl(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(6)
        self.cache.get("a")
        self.clock.advance(6)  # 12s since write
        self.assertIsNone(self.cache.get("a"))

    def test_overwrite_resets_ttl(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(9)
        self.cache.put("a", 2, ttl=10)
        self.clock.advance(9)
        self.assertEqual(2, self.cache.get("a"))

    def test_default_ttl_applies_and_inf_opts_out(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=10, default_ttl=5, clock=self.clock
        )
        cache.put("mortal", 1)
        cache.put("pinned", 2, ttl=math.inf)
        self.clock.advance(5)

        self.assertIsNone(cache.get("mortal"))
        self.assertEqual(2, cache.get("pinned"))

    def test_len_ignores_expired_entries(self) -> None:
        self.cache.put("a", 1, ttl=5)
        self.cache.put("b", 2)
        self.clock.advance(5)
        self.assertEqual(1, len(self.cache))

    def test_purge_expired_reclaims_untouched_keys(self) -> None:
        self.cache.put("a", 1, ttl=5)
        self.cache.put("b", 2, ttl=5)
        self.cache.put("c", 3)
        self.clock.advance(5)

        self.assertEqual(2, self.cache.purge_expired())
        self.assertEqual(0, self.cache.purge_expired())
        self.assertEqual(3, self.cache.get("c"))

    def test_expired_entry_is_sacrificed_before_live_lru(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=2, clock=self.clock)
        cache.put("dying", 1, ttl=5)  # oldest, but about to expire
        cache.put("alive", 2)
        self.clock.advance(5)

        cache.put("new", 3)

        self.assertEqual(2, cache.get("alive"))  # survived despite being LRU
        self.assertEqual(3, cache.get("new"))
        self.assertEqual(0, cache.stats().evictions)
        self.assertEqual(1, cache.stats().expirations)


class TestStats(unittest.TestCase):
    def test_hit_rate(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=2)
        cache.put("a", 1)
        cache.get("a")
        cache.get("a")
        cache.get("b")

        stats = cache.stats()
        self.assertEqual(2, stats.hits)
        self.assertEqual(1, stats.misses)
        self.assertAlmostEqual(2 / 3, stats.hit_rate)

    def test_hit_rate_of_untouched_cache_is_zero(self) -> None:
        self.assertEqual(0.0, InMemoryCache(capacity=1).stats().hit_rate)


class TestConcurrency(unittest.TestCase):
    def test_concurrent_writers_respect_capacity(self) -> None:
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=50)
        barrier = threading.Barrier(8)

        def hammer(offset: int) -> None:
            barrier.wait()
            for i in range(500):
                key = (offset * 500 + i) % 200
                cache.put(key, i)
                cache.get(key)

        threads = [threading.Thread(target=hammer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertLessEqual(len(cache), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
