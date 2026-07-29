# in-memory cache

A fixed-capacity key/value store with per-key TTL and LRU eviction. Get, put,
and eviction are all O(1). Standard library only.

```python
from cache import InMemoryCache

c = InMemoryCache(capacity=1000)
c.put("session:42", {"user": "sohan"}, ttl=30)
c.get("session:42")
```

## Setup

```bash
conda create --name Interview_Practice python=3.12 -y
conda activate Interview_Practice
pip install -r requirements-dev.txt
```

The package needs nothing installed to run; requirements-dev.txt is just pylint
and pytest.

```bash
python demo.py
python -m pytest -q
python -m unittest discover -s tests -t .
```

## API

`InMemoryCache(capacity, *, default_ttl=None, clock=MONOTONIC_CLOCK, eviction_policy=None)`

Capacity must be positive. `default_ttl` applies to any `put` that omits a ttl;
leave it as None for entries that never expire. The clock and the eviction
policy are constructor arguments so they can be swapped in tests or subclasses.

```text
get(key, default=None)      value, or default if missing/expired
put(key, value, ttl=None)   ttl=None uses the cache default, math.inf pins
remove(key)                 True if something was removed
clear()
purge_expired()             drop all dead entries, returns how many
stats()                     CacheStats(hits, misses, evictions, expirations)
capacity                    configured limit
entry_count                 slots held, including dead ones. O(1)
len(cache)                  live entries only. O(n)
key in cache                liveness check, does not affect recency
```

`ExpiryReaper(cache, interval_seconds=60)` runs `purge_expired()` on a
background thread. It is a context manager, so the thread cannot outlive its
scope:

```python
with ExpiryReaper(cache, interval_seconds=30):
    ...
```

`ManualClock` is the test clock. `advance(seconds)` moves virtual time, so TTL
tests need no sleeping:

```python
clock = ManualClock()
c = InMemoryCache(capacity=10, clock=clock)
c.put("k", 1, ttl=10)
clock.advance(10)
assert c.get("k") is None
```

## Structure

```text
cache/
  base.py               Cache interface, CacheStats
  clock.py              Clock protocol, MONOTONIC_CLOCK, ManualClock
  entry.py              CacheEntry: value + expiry deadline
  eviction.py           EvictionPolicy interface, LRUEvictionPolicy
  in_memory_cache.py    InMemoryCache
  reaper.py             ExpiryReaper
tests/test_cache.py
demo.py
```

`InMemoryCache` holds `dict[key, CacheEntry]` and delegates recency to an
eviction policy. The policy only ever sees keys, never values, which is why it
is parameterised on the key type alone. Its whole interface is `on_access`,
`on_remove`, and `candidates`, so an LFU or FIFO variant is a new class rather
than a change to the cache.

`LRUEvictionPolicy` is a dict of keys to nodes in a doubly linked list with
sentinel head and tail:

```text
head <-> a <-> b <-> c <-> tail
          ^                 ^
        coldest          hottest
```

`on_access` unlinks a node and re-appends it next to the tail. The dict finds
the node in constant time and unlinking is two pointer writes, so nothing is
searched. The victim is always `head.next`. The sentinels exist so unlinking
never has to special case the ends.

## Notes on the design

**TTL is stored as an absolute deadline**, computed once at write time, rather
than as a duration. Checking expiry is then one float comparison, and reading an
entry cannot accidentally extend its life. The comparison is `now >= deadline`,
so a zero length TTL is dead immediately rather than briefly live.

**Expiry is lazy.** Nothing runs on a timer by default. An expired entry stays
in the dict until something touches it, at which point it is deleted and the
lookup reports a miss, so an expired key is indistinguishable from one that was
never there.

Lazy expiry is enough for correctness but not for memory: an entry written once
and never read again is never touched, so nothing reclaims it. That is what
`purge_expired` and `ExpiryReaper` are for. The reaper lives outside the cache
and drives it through the public method, so the cache itself has no thread and
no shutdown story, and stays usable in a plain script.

**Eviction prefers dead entries.** Consider capacity 2 holding an expired entry
and a live one. If the expired entry happens to be the newer of the two, plain
LRU would evict the live one and leave the corpse in place. `_make_room` looks
at up to `_EXPIRY_SCAN_LIMIT` keys from the cold end, drops the first expired
one it finds, and only falls back to true LRU if they are all live. The limit
keeps this O(1); scanning the whole cache on every insert would not be.

**The clock is injected.** A hidden call to `time.monotonic` inside the class
would make TTL untestable without real sleeps. The default is `monotonic` rather
than `time.time` because wall clock time can jump backwards (NTP, DST) and
silently extend every TTL in the cache.

**Every key leaves through `_discard`.** It removes the entry, tells the policy,
and updates the counters. Expiry, eviction, and `remove` all route through it,
because the policy's key set and the entry dict have to stay identical. If they
drift, `_make_room` hands back a key that is no longer in the dict.

**`len` and `entry_count` answer different questions.** `len` is what a caller
can actually read, so it skips expired entries and costs O(n). `entry_count` is
how many slots are physically held, which is what matters for memory, and is
O(1).

## Locking

Every public method takes an `RLock`. The private helpers assume the caller
already holds it, which is noted in their docstrings, so there is no nested
acquisition and no lock ordering to get wrong. `get` does its liveness check,
deletion, and promotion inside one acquisition; splitting them would let two
threads both see an entry as live while one deletes it.

This makes the cache one lock wide. For higher throughput the usual next step is
to shard: keep N independent caches and pick one by `hash(key) % N`. That gives
up global LRU in exchange for N-way concurrency, which is normally a fair trade.

## Complexity

```text
get / put / remove / eviction / contains    O(1)
len()                                       O(n)   skips expired entries
entry_count                                 O(1)
purge_expired()                             O(n)   background sweep
```

Space is O(capacity): the entry dict, plus a dict slot and a three field node
per key for recency. `_Node` uses `__slots__` to keep that small.

## Tests

`tests/test_cache.py`, grouped into basic operations, LRU behaviour, TTL
behaviour, stats, and a concurrency check that hammers the cache from eight
threads and asserts capacity holds.

The cases worth knowing about, because they are the rules that are easy to get
wrong:

- `test_membership_check_does_not_affect_recency`: `in` must not promote a key
- `test_reading_does_not_extend_ttl`: deadlines are absolute, not sliding
- `test_put_overwrites_existing_key_without_growing`: overwriting is not an
  insert and must not evict anything
- `test_expired_entry_is_sacrificed_before_live_lru`: covers the scan in
  `_make_room`

## Limitations

- One lock for the whole cache, so throughput is bounded by contention. Sharding
  is the fix.
- `len()` is O(n) because it filters expired entries.
- The linked list is written out by hand, so it is more code than
  `OrderedDict.move_to_end` would need for the same behaviour.
- No protection against a stampede on a hot key expiring; that needs per-key
  locking or early probabilistic refresh.
- Capacity counts entries, not bytes, so one large value can dominate memory.
- No persistence, namespacing, or negative caching.

## Requirements

Python 3.10 or newer (the code uses `dataclass(slots=True)`). Developed on
3.12. If pylint reports `Unable to import cache`, check that it is the pylint
from this environment and not an older one: versions before 3.0 cannot parse
3.12 syntax and report the resulting parse failure as an import error.
