# hsm/core/data_management.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import threading
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Generator

from hsm.core.errors import HSMError


class DataLockError(HSMError):
    """Raised when data lock acquisition fails.

    This might occur if a non-blocking lock attempt is implemented in the future or
    if some unexpected scenario prevents the lock from being acquired.

    Attributes:
        details: Additional context about the locking failure.
    """

    pass


class DataManager:
    """
    Manages state data with thread-safe access.

    Runtime Invariants:
    - State data is stored in a dictionary.
    - All access to mutate or read data should happen within the `access_data()` context.
    - The lock is always held during data access within the context manager block.
    - Snapshots are deep copies, ensuring complete isolation from the original data.

    Example:
        data_manager = DataManager()
        with data_manager.access_data() as data:
            data["counter"] = data.get("counter", 0) + 1
        # Lock is released after context, ensuring thread-safe mutation.

    """

    def __init__(self, lock_timeout: float = 0.1) -> None:
        if lock_timeout < 0:
            raise ValueError("lock_timeout must be non-negative")
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._lock_timeout = lock_timeout

    def _deep_copy_data(self) -> Dict[str, Any]:
        """Create a deep copy of the internal data dictionary."""
        return deepcopy(self._data)

    @contextmanager
    def access_data(self, timeout: float | None = None) -> Generator[Dict[str, Any], None, None]:
        """
        A context manager providing thread-safe access to the internal data dictionary.

        Args:
            timeout: Optional timeout in seconds for lock acquisition.
                    If None, uses the default timeout from initialization.
                    If float('inf'), blocks indefinitely.

        Raises:
            DataLockError: If the lock cannot be acquired within the timeout period.

        Returns:
            A dictionary representing the state data.

        Example:
            with data_manager.access_data() as data:
                data["counter"] = data.get("counter", 0) + 1
        """
        actual_timeout = timeout if timeout is not None else self._lock_timeout

        # Handle infinite timeout specially
        if actual_timeout == float("inf"):
            acquired = self._lock.acquire(blocking=True)
        else:
            acquired = self._lock.acquire(blocking=True, timeout=actual_timeout)

        if not acquired:
            raise DataLockError(f"Failed to acquire data lock within {actual_timeout} seconds.")

        try:
            working_data = self._deep_copy_data()
            yield working_data
            self._data = deepcopy(working_data)
        except Exception:
            raise
        finally:
            self._lock.release()

    def get_data_snapshot(self) -> Dict[str, Any]:
        """
        Return a snapshot (deep copy) of the current state data without locking.

        This method is non-blocking and does not lock the data. It should be used
        only when stale data is acceptable or for debugging. For guaranteed consistency,
        use `access_data()` context manager.

        Returns:
            A deep copy of the current state data, ensuring complete isolation from
            the original data structure including all nested objects.
        """
        return self._deep_copy_data()

    def clear_data(self) -> None:
        """
        Clear all state data in a thread-safe manner.

        Example:
            with data_manager.access_data() as data:
                data.clear()
            # All data is cleared atomically.
        """
        with self.access_data() as data:
            data.clear()
