"""Eviction policies on their own, and LFU driven through a cache."""

# pylint: disable=missing-class-docstring, missing-function-docstring

from __future__ import annotations

import unittest

from cache import InMemoryCache, LFUEvictionPolicy, LRUEvictionPolicy


class TestLruPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy: LRUEvictionPolicy[str] = LRUEvictionPolicy()

    def test_candidates_run_coldest_first(self) -> None:
        for key in "abc":
            self.policy.on_access(key)

        self.assertEqual(["a", "b", "c"], list(self.policy.candidates()))

    def test_access_promotes_a_known_key(self) -> None:
        for key in "abc":
            self.policy.on_access(key)
        self.policy.on_access("a")

        self.assertEqual(["b", "c", "a"], list(self.policy.candidates()))

    def test_evict_candidate_is_the_coldest_key(self) -> None:
        for key in "abc":
            self.policy.on_access(key)

        self.assertEqual("a", self.policy.evict_candidate())

    def test_remove_forgets_the_key(self) -> None:
        for key in "abc":
            self.policy.on_access(key)
        self.policy.on_remove("b")

        self.assertEqual(["a", "c"], list(self.policy.candidates()))
        self.assertEqual(2, len(self.policy))

    def test_removing_an_unknown_key_is_a_no_op(self) -> None:
        self.policy.on_access("a")
        self.policy.on_remove("zzz")

        self.assertEqual(["a"], list(self.policy.candidates()))

    def test_removing_every_key_empties_the_ring(self) -> None:
        for key in "abc":
            self.policy.on_access(key)
        for key in "abc":
            self.policy.on_remove(key)

        self.assertEqual([], list(self.policy.candidates()))
        self.assertIsNone(self.policy.evict_candidate())

    def test_clear(self) -> None:
        for key in "abc":
            self.policy.on_access(key)
        self.policy.clear()

        self.assertEqual([], list(self.policy.candidates()))
        self.assertEqual(0, len(self.policy))

    def test_empty_policy_has_no_candidate(self) -> None:
        self.assertIsNone(self.policy.evict_candidate())

    def test_reaccess_after_removal_reinserts(self) -> None:
        self.policy.on_access("a")
        self.policy.on_remove("a")
        self.policy.on_access("a")

        self.assertEqual(["a"], list(self.policy.candidates()))


class TestLfuPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy: LFUEvictionPolicy[str] = LFUEvictionPolicy()

    def test_least_frequently_used_is_the_victim(self) -> None:
        self.policy.on_access("a")
        self.policy.on_access("b")
        self.policy.on_access("a")

        self.assertEqual("b", self.policy.evict_candidate())

    def test_ties_break_towards_the_older_key(self) -> None:
        self.policy.on_access("a")
        self.policy.on_access("b")

        self.assertEqual("a", self.policy.evict_candidate())
        self.assertEqual(["a", "b"], list(self.policy.candidates()))

    def test_candidates_run_rarest_first(self) -> None:
        self.policy.on_access("rare")
        for _ in range(3):
            self.policy.on_access("common")

        self.assertEqual(["rare", "common"], list(self.policy.candidates()))

    def test_removing_the_only_least_used_key_moves_the_minimum(self) -> None:
        self.policy.on_access("a")
        self.policy.on_access("b")
        self.policy.on_access("b")
        self.policy.on_remove("a")

        self.assertEqual("b", self.policy.evict_candidate())

    def test_a_readded_key_starts_from_zero_again(self) -> None:
        for _ in range(5):
            self.policy.on_access("hot")
        self.policy.on_access("cold")
        self.policy.on_remove("hot")
        self.policy.on_access("hot")

        # "hot" is back on one use, "cold" also has one, and "cold" is older.
        self.assertEqual("cold", self.policy.evict_candidate())

    def test_clear(self) -> None:
        self.policy.on_access("a")
        self.policy.clear()

        self.assertIsNone(self.policy.evict_candidate())
        self.assertEqual(0, len(self.policy))


class TestLfuInsideCache(unittest.TestCase):
    def test_cache_evicts_the_least_used_key(self) -> None:
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=2, eviction_policy=LFUEvictionPolicy()
        )
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.get("a")

        cache.put("c", 3)

        self.assertNotIn("b", cache)
        self.assertEqual(1, cache.get("a"))
        self.assertEqual(3, cache.get("c"))

    def test_frequency_beats_recency(self) -> None:
        # Under LRU "a" would go, because "b" was touched more recently.
        cache: InMemoryCache[str, int] = InMemoryCache(
            capacity=2, eviction_policy=LFUEvictionPolicy()
        )
        cache.put("a", 1)
        for _ in range(5):
            cache.get("a")
        cache.put("b", 2)

        cache.put("c", 3)

        self.assertIn("a", cache)
        self.assertNotIn("b", cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
