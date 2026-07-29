"""Manual walkthrough of the cache: python demo.py"""

import time

from cache import (
    ExpiryReaper,
    InMemoryCache,
    LFUEvictionPolicy,
    ManualClock,
    ShardedCache,
)


def demo_lru_eviction() -> None:
    """A read moves a key out of the eviction slot."""
    print("--- LRU eviction (capacity=2) ---")
    cache: InMemoryCache[str, str] = InMemoryCache(capacity=2)
    cache.put("a", "apple")
    cache.put("b", "banana")

    cache.get("a")  # promotes "a", so "b" is now the eviction candidate
    cache.put("c", "cherry")

    print(f"  a -> {cache.get('a')}")
    print(f"  b -> {cache.get('b')}   (evicted)")
    print(f"  c -> {cache.get('c')}")
    print(f"  {cache.stats()}\n")


def demo_ttl() -> None:
    """Lazy expiry, driven by a virtual clock so nothing has to sleep."""
    print("--- TTL expiry (virtual clock) ---")
    clock = ManualClock()
    cache: InMemoryCache[str, str] = InMemoryCache(capacity=10, clock=clock)
    cache.put("session:42", "sohan", ttl=30)
    cache.put("config", "never-expires")

    clock.advance(29)
    print(f"  t=29s  session:42 -> {cache.get('session:42')}")
    clock.advance(1)
    print(f"  t=30s  session:42 -> {cache.get('session:42')}")
    print(f"  t=30s  config     -> {cache.get('config')}")
    print(f"  {cache.stats()}\n")


def demo_sliding_ttl() -> None:
    """With refresh_ttl_on_access, every read renews the deadline."""
    print("--- Sliding expiry ---")
    clock = ManualClock()
    cache: InMemoryCache[str, str] = InMemoryCache(
        capacity=10, clock=clock, refresh_ttl_on_access=True
    )
    cache.put("token", "abc123", ttl=10)

    for _ in range(3):
        clock.advance(8)
        print(f"  after 8s idle, read -> {cache.get('token')}")

    clock.advance(11)
    print(f"  after 11s idle, read -> {cache.get('token')}\n")


def demo_load_through() -> None:
    """One loader runs per key, however many callers miss at once."""
    print("--- Load-through with single flight ---")
    cache: InMemoryCache[str, str] = InMemoryCache(capacity=10)
    calls = []

    def fetch_user() -> str:
        calls.append(1)
        return "row from the database"

    for _ in range(3):
        cache.get_or_load("user:7", fetch_user)

    print(f"  three get_or_load calls -> loader ran {len(calls)} time(s)")
    print(f"  value -> {cache.get('user:7')}\n")


def demo_removal_listener() -> None:
    """Every departure is reported, with the reason."""
    print("--- Removal listener ---")
    events = []
    cache: InMemoryCache[str, int] = InMemoryCache(
        capacity=2,
        removal_listener=lambda key, value, reason: events.append(
            f"{key}={value} {reason.value}"
        ),
    )
    cache.put("a", 1)
    cache.put("a", 11)  # replaced
    cache.put("b", 2)
    cache.put("c", 3)  # evicts the coldest
    cache.remove("b")

    for event in events:
        print(f"  {event}")
    print()


def demo_weight_limit() -> None:
    """Capacity in bytes rather than entries."""
    print("--- Weight limit ---")
    cache: InMemoryCache[str, str] = InMemoryCache(
        capacity=100, max_weight=20, weigher=lambda key, value: len(value)
    )
    cache.put("small", "x" * 8)
    cache.put("medium", "y" * 8)
    print(f"  weight after two entries: {cache.weight}")
    cache.put("large", "z" * 10)
    print(f"  weight after a third:     {cache.weight}")
    print(f"  survivors: {[k for k in ('small', 'medium', 'large') if k in cache]}\n")


def demo_lfu_policy() -> None:
    """The eviction policy is swappable; nothing in the cache changes."""
    print("--- LFU instead of LRU ---")
    cache: InMemoryCache[str, int] = InMemoryCache(
        capacity=2, eviction_policy=LFUEvictionPolicy()
    )
    cache.put("popular", 1)
    for _ in range(5):
        cache.get("popular")
    cache.put("rare", 2)
    cache.put("new", 3)

    print(f"  popular kept? {'popular' in cache}")
    print(f"  rare kept?    {'rare' in cache}   (LRU would have kept it)\n")


def demo_sharding() -> None:
    """Same interface, several locks."""
    print("--- Sharded cache ---")
    cache: ShardedCache[int, int] = ShardedCache(capacity=1_000, shards=8)
    for i in range(500):
        cache.put(i, i * i)

    print(f"  {cache!r}")
    print(f"  lookup of key 499 -> {cache.get(499)}\n")


def demo_active_expiry() -> None:
    """The reaper reclaims entries that nothing reads back."""
    print("--- Active expiry via reaper (real clock) ---")
    cache: InMemoryCache[str, int] = InMemoryCache(capacity=100)
    for i in range(5):
        cache.put(f"tmp:{i}", i, ttl=0.05)

    with ExpiryReaper(cache, interval_seconds=0.1):
        print(f"  slots held before sweep: {cache.entry_count}")
        time.sleep(0.3)
        print(f"  slots held after sweep:  {cache.entry_count}")


if __name__ == "__main__":
    demo_lru_eviction()
    demo_ttl()
    demo_sliding_ttl()
    demo_load_through()
    demo_removal_listener()
    demo_weight_limit()
    demo_lfu_policy()
    demo_sharding()
    demo_active_expiry()
