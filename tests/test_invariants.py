"""
Invariants the design depends on but that behavioural tests do not cover.

The cache's whole safety argument is that the eviction policy's key set and the
entry dict stay identical, because every removal funnels through one method. That
is asserted directly here, including under concurrent traffic, and the LRU
ordering is checked against an independent reference implementation.

These tests reach into private attributes on purpose: the invariant is internal,
and a test that can only see the public surface cannot check it.
"""

# pylint: disable=missing-class-docstring, missing-function-docstring
# pylint: disable=protected-access

from __future__ import annotations

import unittest
from collections import OrderedDict
from typing import Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from cache import InMemoryCache, ManualClock

from .support import run_concurrently


def assert_consistent(test: unittest.TestCase, cache: InMemoryCache) -> None:
    """The policy must track exactly the keys the cache holds, no more, no less."""
    stored = set(cache._entries)
    tracked = set(cache._policy.candidates())
    test.assertEqual(stored, tracked, "policy and entry dict disagree")
    test.assertLessEqual(len(stored), cache.capacity)


class ReferenceLru:
    """Obvious, slow LRU used only as an oracle."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> Optional[int]:
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key: int, value: int) -> None:
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)

    def remove(self, key: int) -> bool:
        return self.data.pop(key, None) is not None


class TestKeySetInvariant(unittest.TestCase):
    def test_holds_after_ordinary_use(self) -> None:
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=8)
        for i in range(50):
            cache.put(i, i)
            cache.get(i // 2)
            if i % 5 == 0:
                cache.remove(i // 3)

        assert_consistent(self, cache)

    def test_holds_after_expiry_and_eviction_together(self) -> None:
        clock = ManualClock()
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=8, clock=clock)
        for i in range(40):
            cache.put(i, i, ttl=5 if i % 2 else None)
            clock.advance(1)
            cache.get(i - 3)

        cache.purge_expired()
        assert_consistent(self, cache)

    def test_holds_after_clear(self) -> None:
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=8)
        for i in range(20):
            cache.put(i, i)
        cache.clear()

        assert_consistent(self, cache)
        self.assertEqual(0, len(cache))

    def test_holds_under_concurrent_traffic(self) -> None:
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=64)

        def hammer(seed: int) -> None:
            for i in range(800):
                key = (seed * 37 + i) % 200
                choice = i % 4
                if choice == 0:
                    cache.put(key, i)
                elif choice == 1:
                    cache.get(key)
                elif choice == 2:
                    cache.remove(key)
                else:
                    cache.put(key, i, ttl=0.001)

        run_concurrently(hammer)

        assert_consistent(self, cache)

    def test_deadline_heap_does_not_grow_without_bound(self) -> None:
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=4)
        for i in range(5_000):
            cache.put(i % 8, i, ttl=1000)

        # Superseded heap records are compacted, so the heap stays proportional
        # to the number of live entries rather than the number of writes.
        self.assertLess(len(cache._deadlines), 200)


class TestAgainstReferenceImplementation(unittest.TestCase):
    @settings(deadline=None, max_examples=200)
    @given(
        capacity=st.integers(min_value=1, max_value=5),
        operations=st.lists(
            st.one_of(
                st.tuples(
                    st.just("put"),
                    st.integers(min_value=0, max_value=9),
                    st.integers(),
                ),
                st.tuples(
                    st.just("get"), st.integers(min_value=0, max_value=9), st.none()
                ),
                st.tuples(
                    st.just("remove"), st.integers(min_value=0, max_value=9), st.none()
                ),
            ),
            max_size=120,
        ),
    )
    def test_matches_an_ordereddict_lru(
        self,
        capacity: int,
        operations: list[tuple[str, int, Optional[int]]],
    ) -> None:
        cache: InMemoryCache[int, int] = InMemoryCache(capacity=capacity)
        model = ReferenceLru(capacity)

        for name, key, value in operations:
            if name == "put":
                assert value is not None
                cache.put(key, value)
                model.put(key, value)
            elif name == "get":
                self.assertEqual(model.get(key), cache.get(key))
            else:
                self.assertEqual(model.remove(key), cache.remove(key))

            self.assertEqual(set(model.data), set(cache._entries))

        # Same contents and, critically, the same eviction order.
        self.assertEqual(list(model.data), list(cache._policy.candidates()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
