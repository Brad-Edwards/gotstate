# hsm/core/data_management.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import threading
from contextlib import contextmanager
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

    Example:
        data_manager = DataManager()
        with data_manager.access_data() as data:
            data["counter"] = data.get("counter", 0) + 1
        # Lock is released after context, ensuring thread-safe mutation.

    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @contextmanager
    def access_data(self) -> Generator[Dict[str, Any], None, None]:
        """
        A context manager providing thread-safe access to the internal data dictionary.

        Acquires the lock before yielding the data, ensuring that any mutations
        are atomic. Releases the lock after the block exits.

        Raises:
            DataLockError: If the lock cannot be acquired (not expected with a default Lock).

        Returns:
            A dictionary representing the state data.
        """
        acquired = self._lock.acquire(blocking=True)
        if not acquired:
            raise DataLockError("Failed to acquire data lock.")

        try:
            yield self._data
        finally:
            self._lock.release()

    def get_data_snapshot(self) -> Dict[str, Any]:
        """
        Return a snapshot (copy) of the current state data without locking.

        This method is non-blocking and does not lock the data. It should be used
        only when stale data is acceptable or for debugging. For guaranteed consistency,
        use `access_data()` context manager.

        Returns:
            A shallow copy of the current state data.
        """
        # It's safe to read without lock if stale data is acceptable.
        # If strict consistency is needed, user should use `access_data()`.
        return self._data.copy()

    def clear_data(self) -> None:
        """
        Clear all state data in a thread-safe manner.

        Example:
            with data_manager.access_data() as data:
                data.clear()
            # All data is cleared atomically.
        """
        # Instead of doing it outside context, encourage the user to clear within `access_data()`.
        # If we must do it here, we can acquire the lock directly.
        with self.access_data() as data:
            data.clear()
