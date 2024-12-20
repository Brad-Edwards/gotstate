# hsm/runtime/concurrency.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import threading
from contextlib import contextmanager


class _LockFactory:
    """
    Internal factory for producing threading.Lock instances, or potentially other
    concurrency primitives, if needed.
    """

    def create_lock(self) -> threading.Lock:
        """
        Return a new lock instance.
        """
        return threading.Lock()


class _LockContextManager:
    """
    Internal context manager for locking and unlocking a threading.Lock. Used by
    convenience functions that need safe block-level locking.
    """

    def __init__(self, lock: threading.Lock) -> None:
        """
        Store the lock reference.
        """
        self._lock = lock

    def __enter__(self) -> None:
        """
        Acquire the lock on entering the with-block.
        """
        self._lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release the lock on exiting the with-block.
        """
        self._lock.release()


def get_lock() -> threading.Lock:
    """
    Provide a new lock instance to be used for synchronization.
    """
    return _LockFactory().create_lock()


@contextmanager
def with_lock(lock: threading.Lock):
    """
    A convenience context manager that acquires the given lock upon entry and
    releases it upon exit, ensuring safe access to shared resources.
    """
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
