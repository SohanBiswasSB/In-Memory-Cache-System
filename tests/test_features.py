"""Removal listeners, load-through, sliding expiry, and weight limits."""

# pylint: disable=missing-class-docstring, missing-function-docstring

from __future__ import annotations

import threading
import unittest

from cache import InMemoryCache, ManualClock, RemovalReason


class TestRemovalListener(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[tuple[str, int, RemovalReason]] = []
        self.clock = ManualClock()

    def _cache(self, capacity: int = 10, **kwargs: object) -> InMemoryCache[str, int]:
        return InMemoryCache(
            capacity,
            clock=self.clock,
            removal_listener=lambda k, v, r: self.events.append((k, v, r)),
            **kwargs,  # type: ignore[arg-type]
        )

    def test_reports_expiry(self) -> None:
        cache = self._cache()
        cache.put("a", 1, ttl=5)
        self.clock.advance(5)
        cache.get("a")

        self.assertEqual([("a", 1, RemovalReason.EXPIRED)], self.events)

    def test_reports_eviction(self) -> None:
        cache = self._cache(capacity=1)
        cache.put("a", 1)
        cache.put("b", 2)

        self.assertEqual([("a", 1, RemovalReason.EVICTED)], self.events)

    def test_reports_replacement(self) -> None:
        cache = self._cache()
        cache.put("a", 1)
        cache.put("a", 2)

        self.assertEqual([("a", 1, RemovalReason.REPLACED)], self.events)

    def test_reports_explicit_removal(self) -> None:
        cache = self._cache()
        cache.put("a", 1)
        cache.remove("a")

        self.assertEqual([("a", 1, RemovalReason.REMOVED)], self.events)

    def test_reports_every_entry_on_clear(self) -> None:
        cache = self._cache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()

        self.assertEqual(
            {("a", 1, RemovalReason.REMOVED), ("b", 2, RemovalReason.REMOVED)},
            set(self.events),
        )

    def test_reports_entries_reclaimed_by_purge(self) -> None:
        cache = self._cache()
        cache.put("a", 1, ttl=5)
        self.clock.advance(5)
        cache.purge_expired()

        self.assertEqual([("a", 1, RemovalReason.EXPIRED)], self.events)

    def test_failing_listener_does_not_break_the_cache(self) -> None:
        def explode(_k: str, _v: int, _r: RemovalReason) -> None:
            raise RuntimeError("listener is broken")

        cache: InMemoryCache[str, int] = InMemoryCache(1, removal_listener=explode)
        cache.put("a", 1)
        with self.assertLogs("cache.in_memory_cache", level="ERROR"):
            cache.put("b", 2)

        self.assertEqual(2, cache.get("b"))
        self.assertNotIn("a", cache)

    def test_listener_may_call_back_into_the_cache(self) -> None:
        # Listeners run with the lock released, so this must not deadlock.
        seen: list[int] = []
        cache: InMemoryCache[str, int] = InMemoryCache(
            1, removal_listener=lambda k, v, r: seen.append(len(cache))
        )
        cache.put("a", 1)
        cache.put("b", 2)

        self.assertEqual([1], seen)


class TestGetOrLoad(unittest.TestCase):
    def test_loads_on_miss_and_caches_the_result(self) -> None:
        calls = []
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=10)

        def loader() -> int:
            calls.append(1)
            return 42

        self.assertEqual(42, cache.get_or_load("a", loader))
        self.assertEqual(42, cache.get_or_load("a", loader))
        self.assertEqual(1, len(calls))

    def test_caches_a_none_result_rather_than_reloading(self) -> None:
        calls = []
        cache: InMemoryCache[str, None] = InMemoryCache(capacity=10)

        def loader() -> None:
            calls.append(1)

        self.assertIsNone(cache.get_or_load("a", loader))
        self.assertIsNone(cache.get_or_load("a", loader))
        self.assertEqual(1, len(calls))

    def test_honours_ttl(self) -> None:
        clock = ManualClock()
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=10, clock=clock)
        cache.get_or_load("a", lambda: 1, ttl=5)
        clock.advance(5)

        self.assertNotIn("a", cache)

    def test_one_loader_runs_for_concurrent_callers(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=10)
        calls = []
        started = threading.Event()
        results: list[int] = []
        lock = threading.Lock()

        def loader() -> int:
            with lock:
                calls.append(1)
            started.set()
            # Hold long enough that every other thread is queued behind us.
            threading.Event().wait(0.05)
            return 7

        def worker() -> None:
            value = cache.get_or_load("hot", loader)
            with lock:
                results.append(value)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(1, len(calls), "loader should run once, not per caller")
        self.assertEqual([7] * 12, results)

    def test_separate_keys_both_load(self) -> None:
        cache: InMemoryCache[str, str] = InMemoryCache(capacity=10)
        self.assertEqual("a!", cache.get_or_load("a", lambda: "a!"))
        self.assertEqual("b!", cache.get_or_load("b", lambda: "b!"))

    def test_loader_failure_propagates_and_caches_nothing(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=10)

        def loader() -> int:
            raise ValueError("upstream is down")

        with self.assertRaises(ValueError):
            cache.get_or_load("a", loader)
        self.assertNotIn("a", cache)
        # The per-key lock must have been released despite the failure.
        self.assertEqual(1, cache.get_or_load("a", lambda: 1))


class TestSlidingExpiry(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = ManualClock()

    def test_reading_pushes_the_deadline_out(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=10, clock=self.clock, refresh_ttl_on_access=True
        )
        cache.put("a", 1, ttl=10)

        for _ in range(5):
            self.clock.advance(6)
            self.assertEqual(1, cache.get("a"), "each read should renew the TTL")

        self.clock.advance(11)
        self.assertIsNone(cache.get("a"))

    def test_off_by_default(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=10, clock=self.clock)
        cache.put("a", 1, ttl=10)
        self.clock.advance(6)
        cache.get("a")
        self.clock.advance(6)

        self.assertIsNone(cache.get("a"))

    def test_does_not_apply_to_entries_without_a_ttl(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=10, clock=self.clock, refresh_ttl_on_access=True
        )
        cache.put("a", 1)
        self.clock.advance(10_000)

        self.assertEqual(1, cache.get("a"))


class TestWeightLimit(unittest.TestCase):
    @staticmethod
    def _by_length(_key: str, value: str) -> float:
        return float(len(value))

    def test_evicts_until_within_the_weight_limit(self) -> None:
        cache: InMemoryCache[str, str] = InMemoryCache(
            capacity=100, max_weight=10, weigher=self._by_length
        )
        cache.put("a", "x" * 6)
        cache.put("b", "y" * 5)

        self.assertNotIn("a", cache)
        self.assertIn("b", cache)
        self.assertEqual(5, cache.weight)

    def test_evicts_more_than_one_entry_when_needed(self) -> None:
        cache: InMemoryCache[str, str] = InMemoryCache(
            capacity=100, max_weight=10, weigher=self._by_length
        )
        cache.put("a", "x" * 4)
        cache.put("b", "y" * 4)
        cache.put("c", "z" * 9)

        self.assertEqual(["c"], [k for k in ("a", "b", "c") if k in cache])
        self.assertEqual(9, cache.weight)

    def test_weight_defaults_to_entry_count(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(capacity=10)
        cache.put("a", 1)
        cache.put("b", 2)

        self.assertEqual(2, cache.weight)

    def test_replacing_an_entry_adjusts_the_weight(self) -> None:
        cache: InMemoryCache[str, str] = InMemoryCache(
            capacity=100, max_weight=100, weigher=self._by_length
        )
        cache.put("a", "xxx")
        cache.put("a", "y")

        self.assertEqual(1, cache.weight)

    def test_an_entry_heavier_than_the_limit_is_kept_alone(self) -> None:
        cache: InMemoryCache[str, str] = InMemoryCache(
            capacity=100, max_weight=5, weigher=self._by_length
        )
        cache.put("small", "ab")
        cache.put("huge", "x" * 50)

        # Nothing else can help, so the oversized entry stays and is the only one.
        self.assertNotIn("small", cache)
        self.assertEqual("x" * 50, cache.get("huge"))

    def test_weigher_without_max_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryCache(capacity=10, weigher=self._by_length)

    def test_non_positive_max_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryCache(capacity=10, max_weight=0, weigher=self._by_length)

    def test_non_positive_weight_from_weigher_is_rejected(self) -> None:
        cache: InMemoryCache[str, str] = InMemoryCache(
            capacity=10, max_weight=10, weigher=lambda k, v: 0.0
        )
        with self.assertRaises(ValueError):
            cache.put("a", "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
