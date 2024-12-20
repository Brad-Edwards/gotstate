# tests/unit/test_concurrency.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock


def test_lock_factory():
    from hsm.runtime.concurrency import _LockFactory

    lf = _LockFactory()
    lock = lf.create_lock()
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")


def test_lock_context_manager():
    from hsm.runtime.concurrency import _LockContextManager

    lock = MagicMock()
    with _LockContextManager(lock):
        lock.acquire.assert_called_once()
    lock.release.assert_called_once()


def test_get_lock():
    from hsm.runtime.concurrency import get_lock
    
    lock = get_lock()
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")
    

def test_with_lock():
    from hsm.runtime.concurrency import with_lock
    from unittest.mock import MagicMock
    
    lock = MagicMock()
    with with_lock(lock):
        lock.acquire.assert_called_once()
    lock.release.assert_called_once()


def test_with_lock_exception():
    from hsm.runtime.concurrency import with_lock
    from unittest.mock import MagicMock
    
    lock = MagicMock()
    try:
        with with_lock(lock):
            raise Exception("Test error")
    except Exception:
        pass
    # Lock should be released even if exception occurs
    lock.release.assert_called_once()


def test_lock_context_manager_exception():
    from hsm.runtime.concurrency import _LockContextManager
    from unittest.mock import MagicMock
    
    lock = MagicMock()
    manager = _LockContextManager(lock)
    try:
        with manager:
            raise Exception("Test error")
    except Exception:
        pass
    # Verify lock lifecycle
    lock.acquire.assert_called_once()
    lock.release.assert_called_once()
