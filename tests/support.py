"""Helpers shared by the test modules."""

from __future__ import annotations

import threading
import time
from typing import Callable


def run_concurrently(worker: Callable[[int], None], workers: int = 8) -> None:
    """
    Run `worker(index)` in `workers` threads that all start at the same moment.

    The barrier matters: without it the first thread often finishes before the
    last one starts, and the test stops exercising concurrency at all.
    """
    barrier = threading.Barrier(workers)

    def entry(index: int) -> None:
        barrier.wait()
        worker(index)

    threads = [threading.Thread(target=entry, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll until `predicate` holds or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
