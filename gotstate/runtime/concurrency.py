# hsm/runtime/concurrency.py

from __future__ import annotations

import threading
from contextlib import contextmanager


class _LockFactory:
    """Internal factory for producing threading.Lock instances, or other concurrency primitives if needed."""

    def create_lock(self) -> threading.Lock:
        return threading.Lock()


def get_lock() -> threading.Lock:
    """
    Provide a new lock instance to be used for synchronization.
    """
    return _LockFactory().create_lock()


@contextmanager
def with_lock(lock: threading.Lock):
    """
    A convenience context manager that acquires the given lock upon entry
    and releases it upon exit, ensuring safe access to shared resources.
    """
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
