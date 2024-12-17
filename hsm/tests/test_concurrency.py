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


# -----------------------------------------------------------------------------
# LOCK INFO TESTS
# -----------------------------------------------------------------------------
def test_lock_info_immutable() -> None:
    """Test that LockInfo is immutable."""
    info = LockInfo("test", LockState.UNLOCKED)
    with pytest.raises(Exception):
        info.id = "new_id"  # type: ignore


def test_lock_info_defaults() -> None:
    """Test LockInfo default values."""
    info = LockInfo("test", LockState.UNLOCKED)
    assert info.owner is None
    assert info.acquisition_time is None


# -----------------------------------------------------------------------------
# PROTOCOL COMPLIANCE TESTS
# -----------------------------------------------------------------------------
def test_lock_protocol_compliance() -> None:
    """Test that mock lock implements LockProtocol correctly."""
    mock = MockLock("test")
    assert isinstance(mock, LockProtocol)

    # Test protocol methods
    assert mock.acquire()
    assert mock.locked()
    info = mock.get_info()
    assert isinstance(info, LockInfo)
    mock.release()
    assert not mock.locked()


# -----------------------------------------------------------------------------
# INTEGRATION TESTS
# -----------------------------------------------------------------------------
def test_concurrent_lock_access(lock_manager: LockManager) -> None:
    """Test concurrent access to locks."""

    def worker(lock_id: str) -> None:
        lock = lock_manager.create_lock(f"worker_{lock_id}")
        for _ in range(10):
            with lock:
                time.sleep(0.01)

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(lock_manager.get_all_locks()) == 5
    assert all(lock.state != LockState.LOCKED for lock in lock_manager.get_all_locks().values())


@pytest.mark.asyncio
async def test_concurrent_async_lock_access(lock_manager: LockManager) -> None:
    """Test concurrent access to async locks."""

    async def worker(lock_id: str) -> None:
        lock = lock_manager.create_async_lock(f"worker_{lock_id}")
        for _ in range(10):
            async with lock:
                await asyncio.sleep(0.01)

    tasks = [asyncio.create_task(worker(str(i))) for i in range(5)]

    await asyncio.gather(*tasks)

    assert len(lock_manager.get_all_locks()) == 5
    assert all(lock.state != LockState.LOCKED for lock in lock_manager.get_all_locks().values())
