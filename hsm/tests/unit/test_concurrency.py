# hsm/tests/test_concurrency.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time
from typing import Any, Dict, Generator, Optional

import pytest

from hsm.core.errors import HSMError
from hsm.runtime.concurrency import (
    AsyncStateLock,
    ConcurrencyError,
    LockAcquisitionError,
    LockInfo,
    LockManager,
    LockProtocol,
    LockReleaseError,
    LockState,
    StateLock,
)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture
def lock_id() -> str:
    """Fixture providing a unique lock ID."""
    return "test_lock"


@pytest.fixture
def state_lock(lock_id: str) -> StateLock:
    """Fixture providing a StateLock instance."""
    return StateLock(lock_id)


@pytest.fixture
def async_lock(lock_id: str) -> AsyncStateLock:
    """Fixture providing an AsyncStateLock instance."""
    return AsyncStateLock(lock_id)


@pytest.fixture
def lock_manager() -> LockManager:
    """Fixture providing a LockManager instance."""
    return LockManager()


@pytest.fixture
def any_lock(request) -> Generator[LockProtocol, None, None]:
    """Parametrized fixture providing either sync or async lock."""
    lock_type = request.param
    lock = lock_type("test_lock")
    yield lock
    # Cleanup if needed
    if lock.locked():
        lock.release()


@pytest.fixture
def lock_pair() -> Generator[tuple[StateLock, AsyncStateLock], None, None]:
    """Fixture providing both sync and async locks for comparison tests."""
    sync_lock = StateLock("sync")
    async_lock = AsyncStateLock("async")
    yield sync_lock, async_lock
    # Cleanup
    for lock in (sync_lock, async_lock):
        if lock.locked():
            lock.release()


# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------
class MockLock(LockProtocol):
    """Mock lock implementation for testing."""

    def __init__(self, lock_id: str):
        self.id = lock_id
        self._locked = False
        self._owner: Optional[int] = None
        self._acquisition_time: Optional[float] = None

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        self._locked = True
        self._owner = threading.get_ident()
        self._acquisition_time = time.time()
        return True

    def release(self) -> None:
        self._locked = False
        self._owner = None
        self._acquisition_time = None

    def locked(self) -> bool:
        return self._locked

    def get_info(self) -> LockInfo:
        state = LockState.LOCKED if self._locked else LockState.UNLOCKED
        return LockInfo(id=self.id, state=state, owner=self._owner, acquisition_time=self._acquisition_time)


# -----------------------------------------------------------------------------
# ERROR CLASS TESTS
# -----------------------------------------------------------------------------
def test_concurrency_error_inheritance() -> None:
    """Test that ConcurrencyError inherits from HSMError."""
    error = ConcurrencyError("test message")
    assert isinstance(error, HSMError)
    assert str(error) == "test message"


def test_lock_acquisition_error() -> None:
    """Test LockAcquisitionError construction and attributes."""
    lock_id = "test_lock"
    timeout = 1.0
    error = LockAcquisitionError("acquisition failed", lock_id, timeout)
    assert error.lock_id == lock_id
    assert abs(error.timeout - timeout) < 1e-9
    assert error.details == {"lock_id": lock_id, "timeout": timeout}


def test_lock_release_error() -> None:
    """Test LockReleaseError construction and attributes."""
    lock_id = "test_lock"
    owner = 12345
    error = LockReleaseError("release failed", lock_id, owner)
    assert error.lock_id == lock_id
    assert error.owner == owner
    assert error.details == {"lock_id": lock_id, "owner": owner}


# -----------------------------------------------------------------------------
# STATE LOCK TESTS
# -----------------------------------------------------------------------------
def test_state_lock_init(lock_id: str) -> None:
    """Test StateLock initialization."""
    lock = StateLock(lock_id)
    assert not lock.locked()
    assert lock._owner is None
    assert lock._acquisition_time is None


def test_state_lock_init_empty_id() -> None:
    """Test StateLock initialization with empty ID."""
    with pytest.raises(ValueError):
        StateLock("")


def test_state_lock_acquire_release(state_lock: StateLock) -> None:
    """Test basic acquire and release functionality."""
    assert state_lock.acquire()
    assert state_lock.locked()
    assert state_lock._owner == threading.get_ident()
    assert state_lock._acquisition_time is not None

    state_lock.release()
    assert not state_lock.locked()
    assert state_lock._owner is None
    assert state_lock._acquisition_time is None


def test_state_lock_context_manager(state_lock: StateLock) -> None:
    """Test context manager functionality."""
    with state_lock:
        assert state_lock.locked()
        assert state_lock._owner == threading.get_ident()
    assert not state_lock.locked()
    assert state_lock._owner is None


def test_state_lock_release_wrong_thread(state_lock: StateLock) -> None:
    """Test releasing lock from wrong thread."""
    state_lock.acquire()

    def release_in_thread() -> None:
        with pytest.raises(LockReleaseError):
            state_lock.release()

    thread = threading.Thread(target=release_in_thread)
    thread.start()
    thread.join()

    state_lock.release()


def test_state_lock_timeout(state_lock: StateLock) -> None:
    """Test lock acquisition with timeout."""
    state_lock.acquire()

    def acquire_in_thread() -> None:
        with pytest.raises(LockAcquisitionError):
            state_lock.acquire(timeout=0.1)

    thread = threading.Thread(target=acquire_in_thread)
    thread.start()
    thread.join()

    state_lock.release()


def test_state_lock_non_blocking() -> None:
    """Test non-blocking lock acquisition with non-reentrant lock."""
    lock = StateLock("test_lock", reentrant=False)
    lock.acquire()
    assert not lock.acquire(blocking=False)
    lock.release()


def test_state_lock_reentrant() -> None:
    """Test reentrant lock acquisition."""
    lock = StateLock("test_lock", reentrant=True)
    lock.acquire()
    assert lock.acquire(blocking=False)  # Should succeed
    lock.release()
    lock.release()


def test_state_lock_reentrant_count():
    """Test reentrant lock counting"""
    lock = StateLock("test", reentrant=True)
    assert lock._lock_count == 0
    lock.acquire()
    assert lock._lock_count == 1
    lock.acquire()
    assert lock._lock_count == 2
    lock.release()
    assert lock._lock_count == 1


def test_state_lock_owner_tracking():
    """Test lock owner tracking"""
    lock = StateLock("test")
    lock.acquire()
    assert lock._owner == threading.get_ident()
    lock.release()
    assert lock._owner is None


# -----------------------------------------------------------------------------
# ASYNC STATE LOCK TESTS
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_async_lock_init(lock_id: str) -> None:
    """Test AsyncStateLock initialization."""
    lock = AsyncStateLock(lock_id)
    assert not lock.locked()
    assert lock._owner is None
    assert lock._acquisition_time is None


@pytest.mark.asyncio
async def test_async_lock_acquire_release(async_lock: AsyncStateLock) -> None:
    """Test basic async acquire and release functionality."""
    assert await async_lock.acquire()
    assert async_lock.locked()
    assert async_lock._owner == threading.get_ident()
    assert async_lock._acquisition_time is not None

    async_lock.release()
    assert not async_lock.locked()
    assert async_lock._owner is None
    assert async_lock._acquisition_time is None


@pytest.mark.asyncio
async def test_async_lock_context_manager(async_lock: AsyncStateLock) -> None:
    """Test async context manager functionality."""
    async with async_lock:
        assert async_lock.locked()
        assert async_lock._owner == threading.get_ident()
    assert not async_lock.locked()
    assert async_lock._owner is None


@pytest.mark.asyncio
async def test_async_lock_timeout(async_lock: AsyncStateLock) -> None:
    """Test async lock acquisition with timeout."""
    await async_lock.acquire()

    with pytest.raises(LockAcquisitionError):
        await async_lock.acquire(timeout=0.1)

    async_lock.release()


@pytest.mark.asyncio
async def test_async_lock_non_blocking(async_lock: AsyncStateLock) -> None:
    """Test non-blocking async lock acquisition."""
    await async_lock.acquire()
    assert not await async_lock.acquire(blocking=False)
    async_lock.release()


@pytest.mark.asyncio
async def test_async_lock_timeout_zero():
    """Test non-blocking async lock acquisition"""
    lock = AsyncStateLock("test")
    await lock.acquire()
    result = await lock.acquire(timeout=0)
    assert result is False


@pytest.mark.asyncio
async def test_async_lock_owner_change():
    """Test lock owner changes properly"""
    lock = AsyncStateLock("test")
    await lock.acquire()
    owner1 = lock._owner
    lock.release()
    await lock.acquire()
    owner2 = lock._owner
    assert owner1 == owner2  # Same thread


# -----------------------------------------------------------------------------
# LOCK MANAGER TESTS
# -----------------------------------------------------------------------------
def test_lock_manager_create_lock(lock_manager: LockManager, lock_id: str) -> None:
    """Test lock creation through manager."""
    lock = lock_manager.create_lock(lock_id)
    assert isinstance(lock, StateLock)
    assert lock_manager.get_lock(lock_id) is lock


def test_lock_manager_create_async_lock(lock_manager: LockManager, lock_id: str) -> None:
    """Test async lock creation through manager."""
    lock = lock_manager.create_async_lock(lock_id)
    assert isinstance(lock, AsyncStateLock)
    assert lock_manager.get_lock(lock_id) is lock


def test_lock_manager_duplicate_id(lock_manager: LockManager, lock_id: str) -> None:
    """Test creating locks with duplicate IDs."""
    lock_manager.create_lock(lock_id)
    with pytest.raises(ValueError):
        lock_manager.create_lock(lock_id)


def test_lock_manager_remove_lock(lock_manager: LockManager, lock_id: str) -> None:
    """Test lock removal."""
    lock_manager.create_lock(lock_id)
    lock_manager.remove_lock(lock_id)
    assert lock_manager.get_lock(lock_id) is None


def test_lock_manager_remove_locked(lock_manager: LockManager, lock_id: str) -> None:
    """Test removing a locked lock."""
    lock = lock_manager.create_lock(lock_id)
    lock.acquire()
    with pytest.raises(LockReleaseError):
        lock_manager.remove_lock(lock_id)
    lock.release()


def test_lock_manager_get_all_locks(lock_manager: LockManager) -> None:
    """Test retrieving all managed locks."""
    lock1 = lock_manager.create_lock("lock1")
    lock_manager.create_lock("lock2")

    lock1.acquire()
    locks = lock_manager.get_all_locks()

    assert len(locks) == 2
    assert locks["lock1"].state == LockState.LOCKED
    assert locks["lock2"].state == LockState.UNLOCKED

    lock1.release()


def test_lock_manager_mixed_types():
    """Test managing both sync and async locks"""
    manager = LockManager()
    sync_lock = manager.create_lock("sync")
    async_lock = manager.create_async_lock("async")
    assert isinstance(sync_lock, StateLock)
    assert isinstance(async_lock, AsyncStateLock)


def test_lock_manager_cleanup():
    """Test proper cleanup of removed locks"""
    manager = LockManager()
    lock = manager.create_lock("test")
    lock.acquire()
    lock.release()
    manager.remove_lock("test")
    assert "test" not in manager._locks


# Error Cases
async def test_lock_release_unowned():
    """Test releasing an unowned lock"""
    lock = AsyncStateLock("test")
    with pytest.raises(LockReleaseError):
        lock.release()


async def test_lock_double_release():
    """Test releasing a lock twice"""
    lock = StateLock("test")
    lock.acquire()
    lock.release()
    with pytest.raises(LockReleaseError):
        lock.release()


# State Transitions
async def test_lock_state_transitions():
    """Test lock state transitions"""
    lock = StateLock("test")
    assert lock.get_info().state == LockState.UNLOCKED
    lock.acquire()
    assert lock.get_info().state == LockState.LOCKED
    lock.release()
    assert lock.get_info().state == LockState.UNLOCKED


# Edge Cases
@pytest.mark.asyncio
async def test_async_lock_infinite_timeout():
    """Test async lock with infinite timeout"""
    lock = AsyncStateLock("test")
    await lock.acquire()

    async def acquire_with_infinite():
        return await lock.acquire(timeout=None)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(acquire_with_infinite(), timeout=0.1)
    lock.release()


def test_lock_manager_remove_nonexistent():
    """Test removing a non-existent lock"""
    manager = LockManager()
    with pytest.raises(ValueError):
        manager.remove_lock("nonexistent")


@pytest.mark.asyncio
async def test_async_lock_release_during_acquisition():
    """Test releasing lock while another task is waiting to acquire"""
    lock = AsyncStateLock("test")
    await lock.acquire()

    async def delayed_release():
        await asyncio.sleep(0.1)
        lock.release()

    release_task = asyncio.create_task(delayed_release())
    await lock.acquire()  # Should succeed after release
    await release_task
    lock.release()


# Boundary Conditions
def test_state_lock_max_reentrant():
    """Test maximum reentrant lock count"""
    lock = StateLock("test", reentrant=True)
    count = 0
    try:
        while count < 1000:  # Arbitrary large number
            assert lock.acquire(blocking=False)
            count += 1
    finally:
        for _ in range(count):
            lock.release()


@pytest.mark.asyncio
async def test_async_lock_zero_timeout_stress():
    """Test rapid zero-timeout acquisitions"""
    lock = AsyncStateLock("test")
    await lock.acquire()

    results = await asyncio.gather(*[lock.acquire(timeout=0) for _ in range(100)])
    assert all(not result for result in results)
    lock.release()


@pytest.mark.asyncio
async def test_async_lock_rapid_acquire_release():
    """Test rapid acquire/release cycles"""
    lock = AsyncStateLock("test")

    async def acquire_release_cycle():
        for _ in range(10):
            await lock.acquire()
            lock.release()

    # Run multiple cycles concurrently
    await asyncio.gather(*[acquire_release_cycle() for _ in range(10)])


def test_state_lock_stress_reentrant():
    """Test reentrant lock under stress"""
    lock = StateLock("test", reentrant=True)
    max_depth = 100

    def nested_acquire(depth):
        if depth <= 0:
            return
        assert lock.acquire(blocking=False)
        nested_acquire(depth - 1)
        lock.release()

    nested_acquire(max_depth)
    assert not lock.locked()
    assert lock._lock_count == 0


# Helper functions for common test patterns
async def assert_lock_lifecycle(lock: LockProtocol) -> None:
    """Test basic lock lifecycle (acquire, check state, release)."""
    # Initial state
    assert not lock.locked()
    assert lock._owner is None
    assert lock._acquisition_time is None

    # Acquire
    if isinstance(lock, AsyncStateLock):
        await lock.acquire()
    else:
        lock.acquire()

    assert lock.locked()
    assert lock._owner == threading.get_ident()
    assert lock._acquisition_time is not None

    # Release
    lock.release()
    assert not lock.locked()
    assert lock._owner is None
    assert lock._acquisition_time is None


# Replace duplicate tests with parametrized versions
@pytest.mark.parametrize("any_lock", [StateLock, AsyncStateLock], indirect=True)
def test_lock_init(any_lock: LockProtocol) -> None:
    """Test lock initialization for both sync and async locks."""
    assert not any_lock.locked()
    assert any_lock._owner is None
    assert any_lock._acquisition_time is None


@pytest.mark.parametrize("any_lock", [StateLock, AsyncStateLock], indirect=True)
@pytest.mark.asyncio
async def test_lock_owner_tracking(any_lock: LockProtocol) -> None:
    """Test lock owner tracking for both lock types."""
    await assert_lock_lifecycle(any_lock)


@pytest.mark.parametrize("any_lock", [StateLock, AsyncStateLock], indirect=True)
@pytest.mark.asyncio
async def test_lock_release_unowned(any_lock: LockProtocol) -> None:
    """Test releasing an unowned lock for both lock types."""
    with pytest.raises(LockReleaseError):
        any_lock.release()


# Example of combining similar stress tests
@pytest.mark.asyncio
async def test_lock_stress(lock_pair: tuple[StateLock, AsyncStateLock]) -> None:
    """Combined stress test for both lock types."""
    sync_lock, async_lock = lock_pair

    # Test rapid sync operations
    for _ in range(100):
        sync_lock.acquire(blocking=False)
        sync_lock.release()

    # Test rapid async operations
    for _ in range(100):
        if await async_lock.acquire(timeout=0):  # Only release if acquisition was successful
            async_lock.release()
