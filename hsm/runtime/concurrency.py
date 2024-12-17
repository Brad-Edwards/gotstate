# hsm/runtime/concurrency.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, AsyncGenerator, Dict, Generator, Optional, Protocol, TypeVar, runtime_checkable

from hsm.core.errors import HSMError


class ConcurrencyError(HSMError):
    """Base exception for concurrency-related errors.

    Attributes:
        message: Error description
        details: Additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LockAcquisitionError(ConcurrencyError):
    """Raised when a lock cannot be acquired.

    Attributes:
        lock_id: Identifier of the lock that failed
        timeout: Timeout value that was exceeded (if applicable)
    """

    def __init__(self, message: str, lock_id: str, timeout: Optional[float] = None) -> None:
        super().__init__(message, {"lock_id": lock_id, "timeout": timeout})
        self.lock_id = lock_id
        self.timeout = timeout


class LockReleaseError(ConcurrencyError):
    """Raised when a lock cannot be released.

    Attributes:
        lock_id: Identifier of the lock that failed
        owner: Thread ID that owns the lock
    """

    def __init__(self, message: str, lock_id: str, owner: Optional[int] = None) -> None:
        super().__init__(message, {"lock_id": lock_id, "owner": owner})
        self.lock_id = lock_id
        self.owner = owner


class LockState(Enum):
    """Enumeration of possible lock states."""

    UNLOCKED = auto()
    LOCKED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class LockInfo:
    """Information about a lock's current state.

    Attributes:
        id: Unique identifier for the lock
        state: Current state of the lock
        owner: Thread ID of the current owner (if locked)
        acquisition_time: When the lock was acquired (if locked)
    """

    id: str
    state: LockState
    owner: Optional[int] = None
    acquisition_time: Optional[float] = None


@runtime_checkable
class LockProtocol(Protocol):
    """Protocol defining the interface for lock objects.

    Runtime Invariants:
    - Lock state transitions are atomic
    - Only the owner thread can release the lock
    - Lock operations are reentrant
    """

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool: ...
    def release(self) -> None: ...
    def locked(self) -> bool: ...
    def get_info(self) -> LockInfo: ...


T = TypeVar("T")


class StateLock:
    """Thread-safe lock implementation for state machine operations."""

    def __init__(self, lock_id: str, reentrant: bool = True):
        if not lock_id:
            raise ValueError("lock_id cannot be empty")
        self._id = lock_id
        self._reentrant = reentrant
        self._lock = threading.RLock() if reentrant else threading.Lock()
        self._owner: Optional[int] = None
        self._acquisition_time: Optional[float] = None
        self._owner_lock = threading.Lock()
        self._locked = False
        self._lock_count = 0  # Track reentrant lock count

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        try:
            success = self._lock.acquire(blocking=blocking, timeout=timeout if timeout else -1)

            if not success:
                if blocking and timeout is not None:
                    raise LockAcquisitionError(f"Failed to acquire lock {self._id} after {timeout}s", self._id, timeout)
                return False

            with self._owner_lock:
                if not self._locked:
                    self._owner = threading.get_ident()
                    self._acquisition_time = time.time()
                self._locked = True
                self._lock_count += 1
            return True

        except TimeoutError:
            if blocking and timeout is not None:
                raise LockAcquisitionError(f"Failed to acquire lock {self._id} after {timeout}s", self._id, timeout)
            return False

    def release(self) -> None:
        current_thread = threading.get_ident()

        with self._owner_lock:
            if not self._locked:
                return

            if self._owner != current_thread:
                raise LockReleaseError(f"Lock {self._id} cannot be released by non-owner thread", self._id, self._owner)

            try:
                self._lock.release()
                self._lock_count -= 1

                if self._lock_count == 0:
                    self._owner = None
                    self._acquisition_time = None
                    self._locked = False

            except RuntimeError:
                raise LockReleaseError(f"Lock {self._id} cannot be released by non-owner thread", self._id, self._owner)

    def locked(self) -> bool:
        with self._owner_lock:
            return self._locked

    def get_info(self) -> LockInfo:
        with self._owner_lock:
            state = LockState.LOCKED if self._locked else LockState.UNLOCKED
            return LockInfo(id=self._id, state=state, owner=self._owner, acquisition_time=self._acquisition_time)

    def __enter__(self) -> "StateLock":
        self.acquire(blocking=True)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.locked() and self._owner == threading.get_ident():
            self.release()


class AsyncStateLock:
    """Asynchronous lock implementation for state machine operations.

    This lock provides the same functionality as StateLock but for
    async/await code. It uses asyncio primitives internally and
    supports async context manager usage.

    Runtime Invariants:
    - Lock operations are atomic
    - Lock state is consistent across tasks
    - Owner information is accurate
    - Reentrant locking is supported
    - Deadlock prevention via timeout

    Example:
        lock = AsyncStateLock("state_a")
        async with lock:
            # Critical section here
            pass

        # Or explicitly:
        if await lock.acquire(timeout=1.0):
            try:
                # Critical section here
                pass
            finally:
                lock.release()
    """

    def __init__(self, lock_id: str):
        if not lock_id:
            raise ValueError("lock_id cannot be empty")
        self._id = lock_id
        self._lock = asyncio.Lock()
        self._owner: Optional[int] = None
        self._acquisition_time: Optional[float] = None

    async def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire the lock asynchronously.

        Args:
            blocking: If True, block until lock is acquired
            timeout: Maximum time to wait for lock acquisition

        Returns:
            True if lock was acquired, False otherwise

        Raises:
            LockAcquisitionError: If lock cannot be acquired and blocking is True
        """
        try:
            if blocking and timeout is None:
                await self._lock.acquire()
                success = True
            else:
                success = await asyncio.wait_for(
                    self._lock.acquire(), timeout=timeout if timeout else 0 if not blocking else None
                )

            if success:
                self._owner = threading.get_ident()
                self._acquisition_time = time.time()
            return success

        except asyncio.TimeoutError:
            if blocking and timeout is not None:
                raise LockAcquisitionError(f"Failed to acquire lock {self._id} after {timeout}s", self._id, timeout)
            return False

    def release(self) -> None:
        """Release the lock.

        Raises:
            LockReleaseError: If current thread is not the lock owner
        """
        current_thread = threading.get_ident()
        if self._owner != current_thread:
            raise LockReleaseError(f"Lock {self._id} cannot be released by non-owner task", self._id, self._owner)

        self._lock.release()
        self._owner = None
        self._acquisition_time = None

    def locked(self) -> bool:
        """Check if the lock is currently held.

        Returns:
            True if locked, False otherwise
        """
        return self._lock.locked()

    def get_info(self) -> LockInfo:
        """Get current lock information.

        Returns:
            LockInfo object with current state
        """
        if self.locked():
            state = LockState.LOCKED
        else:
            state = LockState.UNLOCKED

        return LockInfo(id=self._id, state=state, owner=self._owner, acquisition_time=self._acquisition_time)

    async def __aenter__(self) -> "AsyncStateLock":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.locked() and self._owner == threading.get_ident():
            self.release()


class LockManager:
    """
    Central manager for lock creation and tracking.

    This class provides a centralized way to create and manage locks,
    ensuring unique lock IDs and proper cleanup.

    Runtime Invariants:
    - Lock IDs are unique
    - Lock lifecycle is properly managed
    - Thread-safe lock creation and cleanup

    Example:
        manager = LockManager()
        lock = manager.create_lock("state_a")
        async_lock = manager.create_async_lock("state_b")
    """

    def __init__(self):
        """Initialize the lock manager."""
        self._locks: Dict[str, LockProtocol] = {}
        self._manager_lock = threading.Lock()

    def create_lock(self, lock_id: str) -> StateLock:
        """Create a new synchronous lock.

        Args:
            lock_id: Unique identifier for the lock

        Returns:
            New StateLock instance

        Raises:
            ValueError: If lock_id already exists
        """
        with self._manager_lock:
            if lock_id in self._locks:
                raise ValueError(f"Lock with id {lock_id} already exists")
            lock = StateLock(lock_id)
            self._locks[lock_id] = lock
            return lock

    def create_async_lock(self, lock_id: str) -> AsyncStateLock:
        """Create a new asynchronous lock.

        Args:
            lock_id: Unique identifier for the lock

        Returns:
            New AsyncStateLock instance

        Raises:
            ValueError: If lock_id already exists
        """
        with self._manager_lock:
            if lock_id in self._locks:
                raise ValueError(f"Lock with id {lock_id} already exists")
            lock = AsyncStateLock(lock_id)
            self._locks[lock_id] = lock
            return lock

    def get_lock(self, lock_id: str) -> Optional[LockProtocol]:
        """Get an existing lock by ID.

        Args:
            lock_id: ID of the lock to retrieve

        Returns:
            The lock if it exists, None otherwise
        """
        return self._locks.get(lock_id)

    def remove_lock(self, lock_id: str) -> None:
        """Remove a lock from management.

        Args:
            lock_id: ID of the lock to remove

        Raises:
            ValueError: If lock_id doesn't exist
            LockReleaseError: If lock is still held
        """
        with self._manager_lock:
            if lock_id not in self._locks:
                raise ValueError(f"Lock with id {lock_id} does not exist")

            lock = self._locks[lock_id]
            if lock.locked():
                raise LockReleaseError(f"Cannot remove locked lock {lock_id}", lock_id, lock.get_info().owner)

            del self._locks[lock_id]

    def get_all_locks(self) -> Dict[str, LockInfo]:
        """Get information about all managed locks.

        Returns:
            Dictionary mapping lock IDs to their current state
        """
        with self._manager_lock:
            return {lock_id: lock.get_info() for lock_id, lock in self._locks.items()}
