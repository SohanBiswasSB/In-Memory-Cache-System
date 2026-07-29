"""
Eviction policies.

A policy tracks key ordering only; the cache owns the values. Keeping them
separate means an LFU or FIFO variant is a new class here rather than a change
to the cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from typing import Generic, Optional, TypeVar

K = TypeVar("K", bound=Hashable)


class EvictionPolicy(ABC, Generic[K]):
    """Decides which key leaves when the cache is at capacity."""

    @abstractmethod
    def on_access(self, key: K) -> None:
        """Record a read or write of `key` (also registers unknown keys)."""

    @abstractmethod
    def on_remove(self, key: K) -> None:
        """Forget `key`; it has left the cache for some reason."""

    @abstractmethod
    def candidates(self) -> Iterator[K]:
        """Keys in eviction order: the best victim first."""

    @abstractmethod
    def clear(self) -> None:
        """Forget all tracked keys."""

    def evict_candidate(self) -> Optional[K]:
        """The single best victim, or None when nothing is tracked."""
        return next(self.candidates(), None)


class _Node(Generic[K]):
    """Link in the recency list. Slotted to keep per-key overhead down."""

    __slots__ = ("key", "prev", "next")

    def __init__(self, key: K) -> None:
        self.key = key
        self.prev: "_Node[K]" = self
        self.next: "_Node[K]" = self


class LRUEvictionPolicy(EvictionPolicy[K]):
    """
    Least recently used, backed by a dict of keys to nodes in a circular doubly
    linked list.

    One sentinel node closes the ring, so `sentinel.next` is the coldest key and
    `sentinel.prev` is the hottest, and unlinking never has to special case the
    ends. All operations are O(1).
    """

    def __init__(self) -> None:
        self._nodes: dict[K, _Node[K]] = {}
        # The sentinel's key is never read; it exists only to anchor the ring.
        self._sentinel: _Node[K] = _Node(None)  # type: ignore[arg-type]

    def on_access(self, key: K) -> None:
        node = self._nodes.get(key)
        if node is None:
            node = _Node(key)
            self._nodes[key] = node
        else:
            self._unlink(node)
        self._append_as_most_recent(node)

    def on_remove(self, key: K) -> None:
        node = self._nodes.pop(key, None)
        if node is not None:
            self._unlink(node)

    def candidates(self) -> Iterator[K]:
        """Walks least-recent to most-recent."""
        node = self._sentinel.next
        while node is not self._sentinel:
            # Snapshot `next` first: the caller may remove the current key.
            following = node.next
            yield node.key
            node = following

    def clear(self) -> None:
        self._nodes.clear()
        self._sentinel.next = self._sentinel
        self._sentinel.prev = self._sentinel

    def __len__(self) -> int:
        return len(self._nodes)

    def _unlink(self, node: _Node[K]) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = node

    def _append_as_most_recent(self, node: _Node[K]) -> None:
        last = self._sentinel.prev
        last.next = node
        node.prev = last
        node.next = self._sentinel
        self._sentinel.prev = node


class LFUEvictionPolicy(EvictionPolicy[K]):
    """
    Least frequently used, with ties broken by which key was touched longest ago.

    Frequency counts live in a dict, and each frequency owns an insertion
    ordered set of the keys at that count. Tracking the smallest live frequency
    keeps eviction O(1) rather than a scan for the minimum.
    """

    def __init__(self) -> None:
        self._counts: dict[K, int] = {}
        self._buckets: dict[int, dict[K, None]] = {}
        self._min_count = 0

    def on_access(self, key: K) -> None:
        count = self._counts.get(key, 0)
        if count:
            self._leave_bucket(key, count)
        self._counts[key] = count + 1
        self._buckets.setdefault(count + 1, {})[key] = None
        if count == 0:
            self._min_count = 1

    def on_remove(self, key: K) -> None:
        count = self._counts.pop(key, None)
        if count is not None:
            self._leave_bucket(key, count)

    def candidates(self) -> Iterator[K]:
        """Walks rarest to most used, oldest first within a frequency."""
        for count in sorted(self._buckets):
            yield from list(self._buckets[count])

    def evict_candidate(self) -> Optional[K]:
        bucket = self._buckets.get(self._min_count)
        if not bucket:
            return next(self.candidates(), None)
        return next(iter(bucket))

    def clear(self) -> None:
        self._counts.clear()
        self._buckets.clear()
        self._min_count = 0

    def __len__(self) -> int:
        return len(self._counts)

    def _leave_bucket(self, key: K, count: int) -> None:
        # Indexed rather than .get(): every counted key is in its bucket, and a
        # KeyError here would mean that invariant had broken silently.
        bucket = self._buckets[count]
        bucket.pop(key, None)
        if bucket:
            return
        del self._buckets[count]
        if count == self._min_count:
            # Nothing is left at the old minimum, so the next count up becomes it.
            self._min_count = min(self._buckets, default=0)
