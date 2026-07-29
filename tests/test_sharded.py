"""ShardedCache: same contract as InMemoryCache, spread over several locks."""

# pylint: disable=missing-class-docstring, missing-function-docstring

from __future__ import annotations

import unittest

from cache import (
    Cache,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    ManualClock,
    ShardedCache,
)

from .support import run_concurrently


class TestShardedCache(unittest.TestCase):
    def test_is_a_cache(self) -> None:
        self.assertIsInstance(ShardedCache(capacity=8, shards=4), Cache)

    def test_stores_and_retrieves_across_shards(self) -> None:
        # Capacity is deliberately far above the key count: keys land by hash, so
        # one shard can fill while others sit idle, and a tight total capacity
        # would evict entries that would have fitted in a single cache.
        cache: ShardedCache[str, int] = ShardedCache(capacity=400, shards=4)
        for i in range(50):
            cache.put(f"k{i}", i)

        for i in range(50):
            self.assertEqual(i, cache.get(f"k{i}"))

    def test_uneven_hashing_can_evict_before_total_capacity_is_reached(self) -> None:
        # The cost of sharding: a busy shard evicts while others have room. This
        # is why sharding trades global LRU for concurrency.
        cache: ShardedCache[int, int] = ShardedCache(capacity=4, shards=4)
        for key in (0, 4, 8, 12):  # hash(int) == int, so all four share a shard
            cache.put(key, key)

        self.assertLess(len(cache), 4)

    def test_a_key_always_lands_in_the_same_shard(self) -> None:
        cache: ShardedCache[str, int] = ShardedCache(capacity=64, shards=8)
        cache.put("stable", 1)
        cache.put("stable", 2)

        self.assertEqual(2, cache.get("stable"))
        self.assertEqual(1, len(cache))

    def test_capacity_is_split_and_rounded_up(self) -> None:
        cache: ShardedCache[str, int] = ShardedCache(capacity=10, shards=4)

        # ceil(10/4) = 3 per shard, so the total is 12 rather than 10.
        self.assertEqual(4, cache.shard_count)
        self.assertEqual(12, cache.capacity)

    def test_total_size_never_exceeds_total_capacity(self) -> None:
        cache: ShardedCache[int, int] = ShardedCache(capacity=8, shards=4)
        for i in range(500):
            cache.put(i, i)

        self.assertLessEqual(len(cache), cache.capacity)

    def test_contains_and_remove(self) -> None:
        cache: ShardedCache[str, int] = ShardedCache(capacity=16, shards=4)
        cache.put("a", 1)

        self.assertIn("a", cache)
        self.assertTrue(cache.remove("a"))
        self.assertFalse(cache.remove("a"))
        self.assertNotIn("a", cache)

    def test_clear_empties_every_shard(self) -> None:
        cache: ShardedCache[int, int] = ShardedCache(capacity=32, shards=4)
        for i in range(20):
            cache.put(i, i)
        cache.clear()

        self.assertEqual(0, len(cache))

    def test_stats_are_totalled_across_shards(self) -> None:
        cache: ShardedCache[str, int] = ShardedCache(capacity=16, shards=4)
        cache.put("a", 1)
        cache.get("a")
        cache.get("missing")

        stats = cache.stats()
        self.assertEqual(1, stats.hits)
        self.assertEqual(1, stats.misses)

    def test_ttl_and_purge_work_across_shards(self) -> None:
        clock = ManualClock()
        cache: ShardedCache[int, int] = ShardedCache(
            capacity=64, shards=4, clock=clock
        )
        for i in range(20):
            cache.put(i, i, ttl=5)
        clock.advance(5)

        self.assertEqual(20, cache.purge_expired())
        self.assertEqual(0, len(cache))

    def test_get_or_load(self) -> None:
        cache: ShardedCache[str, int] = ShardedCache(capacity=16, shards=4)
        calls = []

        def loader() -> int:
            calls.append(1)
            return 5

        self.assertEqual(5, cache.get_or_load("a", loader))
        self.assertEqual(5, cache.get_or_load("a", loader))
        self.assertEqual(1, len(calls))

    def test_concurrent_traffic_stays_within_capacity(self) -> None:
        cache: ShardedCache[int, int] = ShardedCache(capacity=64, shards=8)

        def hammer(seed: int) -> None:
            for i in range(500):
                key = (seed * 500 + i) % 300
                cache.put(key, i)
                cache.get(key)

        run_concurrently(hammer)

        self.assertLessEqual(len(cache), cache.capacity)

    def test_rejects_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            ShardedCache(capacity=0, shards=4)
        with self.assertRaises(ValueError):
            ShardedCache(capacity=10, shards=0)

    def test_rejects_a_shared_policy_instance(self) -> None:
        # One policy across shards would track every key, so a shard could be
        # told to evict a key it does not hold.
        with self.assertRaises(ValueError) as caught:
            ShardedCache(capacity=8, shards=4, eviction_policy=LRUEvictionPolicy())

        self.assertIn("policy_factory", str(caught.exception))

    def test_policy_factory_gives_each_shard_its_own_policy(self) -> None:
        cache: ShardedCache[int, int] = ShardedCache(
            capacity=8, shards=4, policy_factory=LFUEvictionPolicy
        )
        # This pattern is what breaks with a shared policy: key 1 lands in one
        # shard and is globally coldest, while another shard is the one evicting.
        cache.put(1, 1)
        for key in (0, 4, 8, 12, 16):
            cache.put(key, key)

        self.assertEqual(1, cache.get(1))
        self.assertLessEqual(len(cache), cache.capacity)

    def test_repr_mentions_shape(self) -> None:
        cache: ShardedCache[str, int] = ShardedCache(capacity=8, shards=2)

        self.assertIn("shards=2", repr(cache))


if __name__ == "__main__":
    unittest.main(verbosity=2)
