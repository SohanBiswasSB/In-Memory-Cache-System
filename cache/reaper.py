"""Optional active expiry: a daemon thread that sweeps dead entries."""

from __future__ import annotations

import threading
from typing import Optional

from .in_memory_cache import InMemoryCache


class ExpiryReaper:
    """
    Calls cache.purge_expired() on an interval.

    Lazy expiry never serves a stale value, but it also never reclaims a key
    that nothing reads again. This thread does that reclaiming.

    Use as a context manager so the thread is always stopped:

        with ExpiryReaper(cache, interval_seconds=30):
            ...
    """

    def __init__(self, cache: InMemoryCache, interval_seconds: float = 60.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._cache = cache
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ExpiryReaper":
        """Start the sweep thread. Returns self so the call can be chained."""
        if self._thread is not None:
            raise RuntimeError("reaper already started")
        self._thread = threading.Thread(
            target=self._run, name="cache-expiry-reaper", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """Signal the thread and wait for it to finish. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        # wait() is both the sleep and the shutdown check, so stop() takes effect
        # immediately instead of after the rest of the interval.
        while not self._stop.wait(self._interval):
            self._cache.purge_expired()

    def __enter__(self) -> "ExpiryReaper":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
