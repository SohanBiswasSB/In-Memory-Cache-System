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

    def __init__(self, key: Optional[K] = None) -> None:
        self.key = key
        self.prev: Optional["_Node[K]"] = None
        self.next: Optional["_Node[K]"] = None


class LRUEvictionPolicy(EvictionPolicy[K]):
    """
    Least recently used, backed by a dict of keys to nodes in a doubly linked
    list.

    Layout is head <-> least recent ... most recent <-> tail. Head and tail are
    sentinels so unlinking never has to special case the ends. All operations
    are O(1).
    """

    def __init__(self) -> None:
        self._nodes: dict[K, _Node[K]] = {}
        self._head: _Node[K] = _Node()  # LRU side
        self._tail: _Node[K] = _Node()  # MRU side
        self._head.next = self._tail
        self._tail.prev = self._head

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
        node = self._head.next
        while node is not self._tail:
            assert node is not None and node.key is not None
            # Snapshot `next` first: the caller may remove the current key.
            following = node.next
            yield node.key
            node = following

    def clear(self) -> None:
        self._nodes.clear()
        self._head.next = self._tail
        self._tail.prev = self._head

    def __len__(self) -> int:
        return len(self._nodes)

    # --- list plumbing -------------------------------------------------

    def _unlink(self, node: _Node[K]) -> None:
        node.prev.next = node.next  # type: ignore[union-attr]
        node.next.prev = node.prev  # type: ignore[union-attr]
        node.prev = node.next = None

    def _append_as_most_recent(self, node: _Node[K]) -> None:
        last = self._tail.prev
        last.next = node  # type: ignore[union-attr]
        node.prev = last
        node.next = self._tail
        self._tail.prev = node
