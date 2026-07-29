# in-memory cache

![ci](https://github.com/SohanBiswasSB/In-Memory-Cache-System/actions/workflows/ci.yml/badge.svg)

A fixed-capacity key/value store with per-key TTL and LRU eviction. Get, put,
and eviction are all O(1). Standard library only.

```python
from cache import InMemoryCache

c = InMemoryCache(capacity=1000)
c.put("session:42", {"user": "sohan"}, ttl=30)
c.get("session:42")
```

Beyond the basics it also does load-through with single flight, removal
listeners, sliding expiry, weight-based limits, pluggable eviction policies, and
sharding.

## Setup

```bash
conda create --name Interview_Practice python=3.12 -y
conda activate Interview_Practice
pip install -e ".[dev]"
```

The package itself has no dependencies. The dev extras are pylint, pytest,
pytest-cov, mypy and hypothesis.

```bash
python demo.py                                  # narrated walkthrough
python benchmark.py                             # throughput numbers
pytest -q --cov=cache --cov-report=term-missing
pylint cache tests demo.py benchmark.py
mypy
```

## API

`InMemoryCache(capacity, *, ...)`, all options keyword-only:

```text
capacity                max entries, must be positive
default_ttl             seconds applied when put() omits a ttl
clock                   any () -> float, injected for testing
eviction_policy         LRUEvictionPolicy() by default, or LFU, or your own
refresh_ttl_on_access   renew the deadline on every read (sliding expiry)
max_weight              cap total weight instead of just entry count
weigher                 (key, value) -> float, required with max_weight
removal_listener        (key, value, reason) -> None, called on every departure
```

```text
get(key, default=None)        value, or default if missing/expired
get_or_load(key, loader, ttl) cached value, or load it once and store it
put(key, value, ttl=None)     ttl=None uses the cache default, math.inf pins
remove(key)                   True if something was removed
clear()
purge_expired(batch=512)      drop dead entries, returns how many
stats()                       CacheStats(hits, misses, evictions, expirations)
capacity                      configured limit
entry_count                   slots held, including dead ones. O(1)
weight                        total weight held
len(cache)                    live entries only. O(n)
key in cache                  liveness check, does not affect recency
```

`RemovalReason` is one of `EXPIRED`, `EVICTED`, `REMOVED`, `REPLACED`.

`ExpiryReaper(cache, interval_seconds=60)` runs `purge_expired()` on a background
thread. It is a context manager, so the thread cannot outlive its scope:

```python
with ExpiryReaper(cache, interval_seconds=30):
    ...
```

`ShardedCache(capacity, *, shards=8, **cache_options)` implements the same
interface across several independent caches. `ManualClock` is the test clock;
`advance(seconds)` moves virtual time so TTL tests need no sleeping:

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
  base.py               Cache interface, CacheStats, RemovalReason
  clock.py              Clock protocol, MONOTONIC_CLOCK, ManualClock
  entry.py              CacheEntry: value, deadline, ttl, weight
  eviction.py           EvictionPolicy interface, LRU and LFU policies
  in_memory_cache.py    InMemoryCache
  sharded.py            ShardedCache
  reaper.py             ExpiryReaper
tests/                  108 tests
demo.py
benchmark.py
```

`InMemoryCache` holds `dict[key, CacheEntry]` and delegates recency to an
eviction policy. The policy only ever sees keys, never values, which is why it is
parameterised on the key type alone. Its whole interface is `on_access`,
`on_remove`, and `candidates`, so LFU is a separate class rather than a change to
the cache, and so is anything else you want to write.

`LRUEvictionPolicy` is a dict of keys to nodes in a circular doubly linked list
with a single sentinel:

```text
sentinel <-> a <-> b <-> c <-> back to sentinel
              ^           ^
            coldest     hottest
```

`on_access` unlinks a node and re-appends it before the sentinel. The dict finds
the node in constant time and unlinking is two pointer writes, so nothing is
searched. The victim is always `sentinel.next`. Closing the ring means unlinking
never has to check for the ends, and no node field is ever None.

Deadlines live in a separate min-heap of `(deadline, tiebreak, key)`, so expired
keys can be found without scanning the entries. The tiebreak counter keeps tuple
comparison away from the keys, which do not have to be orderable.

## Notes on the design

**TTL is stored as an absolute deadline**, computed once at write time, rather
than as a duration. Checking expiry is then one float comparison, and reading an
entry cannot accidentally extend its life. The comparison is `now >= deadline`,
so a zero length TTL is dead immediately rather than briefly live.

**Expiry is lazy.** Nothing runs on a timer by default. An expired entry stays in
the dict until something touches it, at which point it is deleted and the lookup
reports a miss, so an expired key is indistinguishable from one that was never
there.

Lazy expiry is enough for correctness but not for memory: an entry written once
and never read again is never touched, so nothing reclaims it. That is what
`purge_expired` and `ExpiryReaper` are for. Because the sweep is driven by the
deadline heap rather than a scan, it costs time proportional to the number of
dead keys rather than the size of the cache, and it releases the lock between
batches so a large sweep cannot stall readers for its whole duration.

The heap holds a record per write, not per key, so a key written repeatedly
leaves records behind. Those are recognised when popped, by comparing the record
against the entry's current deadline, and the heap is rebuilt once the stale
records outnumber the live entries. Removing them on write instead would cost
O(n) per write.

**Eviction prefers dead entries.** Given a cache holding one expired entry and
one live one, plain LRU could evict the live one and leave the corpse in place.
Eviction pops the heap first, so anything already expired goes before anything
that is still valid.

**Load-through uses one loader per key.** `get_or_load` takes a per-key lock,
re-checks the cache after acquiring it, and only then calls the loader. Without
that, a hot key expiring lets every concurrent caller hit the database at once.
The loader runs with the cache lock released, and different keys never block each
other.

**Removal listeners run outside the lock.** They are queued while the lock is
held and delivered after it is released, so a listener can call back into the
cache without deadlocking, and cannot hold up other threads. An exception from a
listener is logged and swallowed, because a broken listener should not take the
cache down.

**The clock is injected.** A hidden call to `time.monotonic` inside the class
would make TTL untestable without real sleeps. The default is `monotonic` rather
than `time.time` because wall clock time can jump backwards (NTP, DST) and
silently extend every TTL in the cache.

**Every key leaves through `_discard`.** It removes the entry, tells the policy,
adjusts the weight, and queues the listener event. Expiry, eviction, and `remove`
all route through it, because the policy's key set and the entry dict have to
stay identical. There is a test that asserts exactly that, including under
concurrent traffic.

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

That makes a single cache one lock wide. `ShardedCache` is the way out: N
independent caches chosen by `hash(key) % N`, so N threads can work at once. The
cost is that LRU becomes per-shard, and a busy shard evicts while other shards
still have room. There is a test that demonstrates exactly that, because it is a
real cost rather than a footnote.

## Complexity

```text
get / put / remove / eviction / contains    O(1)
get_or_load, cached                         O(1)
purge_expired                               O(k log n) in the number of dead keys
len()                                       O(n)   skips expired entries
entry_count / weight                        O(1)
```

Space is O(capacity): the entry dict, a dict slot and a three field node per key
for recency, and a heap record per key with a TTL.

`python benchmark.py` measures it. Numbers from one laptop run, so treat them as
shape rather than spec:

```text
put, never evicting            166,000 ops/sec
put, evicting every time       109,000 ops/sec
get, hit                       439,000 ops/sec
get, miss                      842,000 ops/sec
purge_expired                  317,000 entries/sec

lookup rate as the cache grows
    1,000 entries              504,000 ops/sec
   10,000 entries              468,000 ops/sec
  100,000 entries              406,000 ops/sec
1,000,000 entries              394,000 ops/sec
```

The last block is the point: a thousandfold increase in size costs about 20% in
lookup rate, which is memory locality rather than algorithmic growth.

## Tests

108 tests, 100% line coverage of `cache/`, checked by pylint (10.00/10) and mypy
in strict mode.

```text
test_cache.py        basic operations, LRU behaviour, TTL behaviour, stats
test_features.py     removal listeners, load-through, sliding expiry, weights
test_policies.py     LRU and LFU policies on their own, and inside a cache
test_sharded.py      routing, capacity split, totals, the sharding trade-off
test_reaper.py       start/stop/restart, sweeping, surviving a failing sweep
test_invariants.py   the key set invariant, and a property-based oracle
test_edges.py        paths ordinary use does not reach
```

Two of these are worth more than the rest. `test_invariants.py` asserts that the
policy's key set and the entry dict stay identical, including after eight threads
hammer the cache with mixed operations, which is the assumption the whole design
rests on. It also replays random operation sequences against an `OrderedDict`
reference implementation and compares both contents and eviction order, which
catches ordering bugs that hand-written cases miss.

The cases that pin down the easy-to-break rules:

- `test_membership_check_does_not_affect_recency`: `in` must not promote a key
- `test_reading_does_not_extend_ttl`: deadlines are absolute unless you ask for
  sliding expiry
- `test_put_overwrites_existing_key_without_growing`: overwriting is not an
  insert and must not evict anything
- `test_one_loader_runs_for_concurrent_callers`: twelve threads, one loader call
- `test_the_entry_just_written_is_never_the_victim`: reachable under LFU, where a
  brand new key has the lowest use count

## Limitations

- A single cache is one lock wide, so throughput is bounded by contention.
  `ShardedCache` trades global LRU for concurrency.
- `len()` is O(n) because it filters expired entries.
- Weight limits are advisory in one case: an entry heavier than the whole limit
  is kept rather than rejected, after everything else has been evicted.
- The linked list is written out by hand, so it is more code than
  `OrderedDict.move_to_end` would need for the same behaviour.
- No persistence, key namespacing, or negative caching.
- `stats()` on a sharded cache is summed shard by shard, so it is not a single
  instant.

## Requirements

Python 3.10 or newer (the code uses `dataclass(slots=True)`). CI runs 3.10, 3.11
and 3.12. If pylint reports `Unable to import cache`, check that it is the pylint
from this environment and not an older one: versions before 3.0 cannot parse 3.12
syntax and report the resulting parse failure as an import error.
