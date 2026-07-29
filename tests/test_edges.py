"""Edge cases and the paths that ordinary use does not reach."""

# pylint: disable=missing-class-docstring, missing-function-docstring

from __future__ import annotations

import unittest
from collections.abc import Iterator

from cache import (
    CacheEntry,
    CacheStats,
    EvictionPolicy,
    InMemoryCache,
    LFUEvictionPolicy,
    ManualClock,
)


class _TracksNothingPolicy(EvictionPolicy[str]):
    """A policy that never nominates a victim, so eviction cannot happen."""

    def on_access(self, key: str) -> None:
        pass

    def on_remove(self, key: str) -> None:
        pass

    def candidates(self) -> Iterator[str]:
        return iter(())

    def clear(self) -> None:
        pass


class TestCacheStats(unittest.TestCase):
    def test_adding_totals_each_counter(self) -> None:
        left = CacheStats(hits=1, misses=2, evictions=3, expirations=4)
        right = CacheStats(hits=10, misses=20, evictions=30, expirations=40)

        self.assertEqual(CacheStats(11, 22, 33, 44), left + right)

    def test_adding_a_non_stats_object_is_not_supported(self) -> None:
        with self.assertRaises(TypeError):
            _ = CacheStats() + 5  # type: ignore[operator]


class TestManualClock(unittest.TestCase):
    def test_refuses_to_run_backwards(self) -> None:
        clock = ManualClock(start=10.0)
        with self.assertRaises(ValueError):
            clock.advance(-1)

    def test_starts_where_told(self) -> None:
        self.assertEqual(7.5, ManualClock(start=7.5)())


class TestCacheEntry(unittest.TestCase):
    def test_renewing_an_entry_without_a_ttl_returns_it_unchanged(self) -> None:
        entry = CacheEntry.immortal("value")

        self.assertIs(entry, entry.renewed(now=100.0))

    def test_renewing_pushes_the_deadline_out_by_the_original_ttl(self) -> None:
        entry = CacheEntry.expiring_at("value", expires_at=10.0, ttl=10.0)
        renewed = entry.renewed(now=8.0)

        self.assertEqual(18.0, renewed.expires_at)
        self.assertEqual(10.0, renewed.ttl)

    def test_a_zero_length_lifetime_is_never_live(self) -> None:
        entry = CacheEntry.expiring_at("value", expires_at=5.0)

        self.assertTrue(entry.is_expired(5.0))


class TestEvictionEdgeCases(unittest.TestCase):
    def test_a_policy_that_tracks_nothing_disables_eviction(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=1, eviction_policy=_TracksNothingPolicy()
        )
        cache.put("a", 1)
        cache.put("b", 2)

        # Nothing can be nominated, so both entries stay and capacity is exceeded.
        self.assertEqual(2, cache.entry_count)
        self.assertEqual(0, cache.stats().evictions)

    def test_the_entry_just_written_is_never_the_victim(self) -> None:
        # LFU makes this reachable: a brand new key has the lowest use count, so
        # it is the natural victim, but evicting the write that triggered the
        # eviction would be absurd.
        cache: InMemoryCache[str, str] = InMemoryCache(
            capacity=100,
            max_weight=10,
            weigher=lambda key, value: float(len(value)),
            eviction_policy=LFUEvictionPolicy(),
        )
        cache.put("old", "x" * 6)
        for _ in range(3):
            cache.get("old")

        cache.put("new", "y" * 6)

        self.assertEqual("y" * 6, cache.get("new"))
        self.assertNotIn("old", cache)


class _PhantomVictimPolicy(EvictionPolicy[str]):
    """Nominates a key the cache never held, simulating policy drift."""

    def on_access(self, key: str) -> None:
        pass

    def on_remove(self, key: str) -> None:
        pass

    def candidates(self) -> Iterator[str]:
        return iter(("ghost",))

    def clear(self) -> None:
        pass


class TestPolicyDrift(unittest.TestCase):
    def test_a_policy_nominating_an_unknown_key_fails_loudly(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=1, eviction_policy=_PhantomVictimPolicy()
        )
        cache.put("a", 1)

        with self.assertRaises(RuntimeError) as caught:
            cache.put("b", 2)

        message = str(caught.exception)
        self.assertIn("ghost", message)
        self.assertIn("does not hold", message)


class TestRepr(unittest.TestCase):
    def test_repr_reports_size_capacity_and_policy(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4)
        cache.put("a", 1)

        text = repr(cache)
        self.assertIn("size=1", text)
        self.assertIn("capacity=4", text)
        self.assertIn("LRUEvictionPolicy", text)


class TestDefaults(unittest.TestCase):
    def test_get_returns_the_supplied_default_for_an_expired_key(self) -> None:
        clock = ManualClock()
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=4, clock=clock)
        cache.put("a", 1, ttl=5)
        clock.advance(5)

        self.assertEqual(-1, cache.get("a", default=-1))

    def test_default_ttl_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryCache(capacity=4, default_ttl=0)

    def test_unhashable_keys_are_rejected_by_the_dict(self) -> None:
        cache: InMemoryCache[object, int] = InMemoryCache(capacity=4)
        with self.assertRaises(TypeError):
            cache.put(["not", "hashable"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
