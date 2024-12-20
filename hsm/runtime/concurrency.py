# hsm/runtime/concurrency.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading


class _LockFactory:
    """
    Internal factory for producing threading.Lock instances, or potentially other
    concurrency primitives, if needed.
    """

    def create_lock(self) -> "threading.Lock":
        """
        Return a new lock instance.
        """
        raise NotImplementedError()


class _LockContextManager:
    """
    Internal context manager for locking and unlocking a threading.Lock. Used by
    convenience functions that need safe block-level locking.
    """

    def __init__(self, lock: "threading.Lock") -> None:
        """
        Store the lock reference.
        """
        raise NotImplementedError()

    def __enter__(self) -> None:
        """
        Acquire the lock on entering the with-block.
        """
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release the lock on exiting the with-block.
        """
        raise NotImplementedError()
