# hsm/core/data_management.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import threading
from contextlib import contextmanager


class _DataLockManager:
    """
    Internal lock manager controlling access to state data. Ensures thread-safe
    reads and writes.
    """

    def __init__(self) -> None:
        """
        Initialize internal lock structures.
        """
        self._lock = threading.Lock()

    def lock(self) -> None:
        """
        Acquire the data lock.
        """
        self._lock.acquire()

    def unlock(self) -> None:
        """
        Release the data lock.
        """
        self._lock.release()


class _ScopedDataContext:
    """
    Internal context manager used to lock and unlock state data within a 'with'
    block, providing a safe scope for data manipulations.
    """

    def __init__(self, lock_manager: _DataLockManager) -> None:
        """
        Store reference to the lock manager.
        """
        self._lock_manager = lock_manager

    def __enter__(self) -> None:
        """
        Acquire the lock when entering the context.
        """
        self._lock_manager.lock()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release the lock when exiting the context.
        """
        self._lock_manager.unlock()


def with_state_data_lock(state: "State"):
    """
    Public API function that returns a context manager for locking state data.
    Uses _DataLockManager internally. If each state maintains its own lock, we
    can store it as state._lock_manager and use it here.
    For simplicity, assume each state can have a lock manager attribute.
    If not present, create one on-demand.
    """
    if not hasattr(state, "_lock_manager"):
        state._lock_manager = _DataLockManager()
    return _ScopedDataContext(state._lock_manager)
