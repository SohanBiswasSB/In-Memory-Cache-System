"""Manual walkthrough of the cache: python demo.py"""

import time

from cache import ExpiryReaper, InMemoryCache, ManualClock


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
    demo_active_expiry()
