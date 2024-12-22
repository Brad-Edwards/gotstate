# tests/unit/test_concurrency.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock

from hsm.runtime.concurrency import get_lock, with_lock


def test_lock_factory():
    """Test that the lock factory creates proper locks."""
    from hsm.runtime.concurrency import _LockFactory

    lf = _LockFactory()
    lock = lf.create_lock()
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")


def test_get_lock():
    """Test that get_lock returns a proper lock."""
    lock = get_lock()
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")


def test_with_lock():
    """Test that with_lock properly acquires and releases the lock."""
    lock = MagicMock()
    with with_lock(lock):
        lock.acquire.assert_called_once()
    lock.release.assert_called_once()


def test_with_lock_exception():
    """Test that with_lock releases the lock even when an exception occurs."""
    lock = MagicMock()
    try:
        with with_lock(lock):
            raise Exception("Test error")
    except Exception:
        pass
    # Lock should be released even if exception occurs
    lock.acquire.assert_called_once()
    lock.release.assert_called_once()


def test_with_lock_nested():
    """Test that with_lock works correctly with nested locks."""
    lock1 = MagicMock()
    lock2 = MagicMock()

    with with_lock(lock1):
        lock1.acquire.assert_called_once()
        with with_lock(lock2):
            lock2.acquire.assert_called_once()
        lock2.release.assert_called_once()
    lock1.release.assert_called_once()
