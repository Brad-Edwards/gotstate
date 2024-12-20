# hsm/core/data_management.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
class _DataLockManager:
    """
    Internal lock manager controlling access to state data. Ensures thread-safe
    reads and writes.
    """

    def __init__(self) -> None:
        """
        Initialize internal lock structures.
        """
        raise NotImplementedError()

    def lock(self) -> None:
        """
        Acquire the data lock.
        """
        raise NotImplementedError()

    def unlock(self) -> None:
        """
        Release the data lock.
        """
        raise NotImplementedError()


class _ScopedDataContext:
    """
    Internal context manager used to lock and unlock state data within a 'with'
    block, providing a safe scope for data manipulations.
    """

    def __init__(self, lock_manager: _DataLockManager) -> None:
        """
        Store reference to the lock manager.
        """
        raise NotImplementedError()

    def __enter__(self) -> None:
        """
        Acquire the lock when entering the context.
        """
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release the lock when exiting the context.
        """
        raise NotImplementedError()
