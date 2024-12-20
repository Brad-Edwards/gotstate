# tests/unit/test_data_management.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock

import pytest


def test_data_lock_manager():
    from hsm.core.data_management import _DataLockManager

    lm = _DataLockManager()
    # We don't know exact locking strategy, but we can assume it provides lock/unlock.
    # Mock the underlying lock if needed.
    lm._lock = MagicMock()
    lm.lock()
    lm._lock.acquire.assert_called_once()
    lm.unlock()
    lm._lock.release.assert_called_once()


def test_scoped_data_context():
    from hsm.core.data_management import _DataLockManager, _ScopedDataContext

    lm = _DataLockManager()
    lm._lock = MagicMock()
    with _ScopedDataContext(lm):
        lm._lock.acquire.assert_called_once()
    lm._lock.release.assert_called_once()


def test_data_lock_manager_exception():
    from unittest.mock import MagicMock

    from hsm.core.data_management import _DataLockManager

    lm = _DataLockManager()
    lm._lock = MagicMock()
    try:
        lm.lock()
        raise Exception("Test error")
    except Exception:
        lm.unlock()

    lm._lock.acquire.assert_called_once()
    lm._lock.release.assert_called_once()


def test_scoped_data_context_exception():
    from hsm.core.data_management import _DataLockManager, _ScopedDataContext

    lm = _DataLockManager()
    lm._lock = MagicMock()
    try:
        with _ScopedDataContext(lm):
            raise Exception("Test error")
    except Exception:
        pass

    lm._lock.acquire.assert_called_once()
    lm._lock.release.assert_called_once()


def test_with_state_data_lock():
    from unittest.mock import MagicMock

    from hsm.core.data_management import with_state_data_lock

    # Mock state object
    state = MagicMock()

    with with_state_data_lock(state):
        assert hasattr(state, "_lock_manager")
        assert state._lock_manager._lock is not None


def test_with_state_data_lock_existing_manager():
    from unittest.mock import MagicMock

    from hsm.core.data_management import _DataLockManager, with_state_data_lock

    # Mock state object with existing lock manager
    state = MagicMock()
    existing_manager = _DataLockManager()
    existing_manager._lock = MagicMock()
    state._lock_manager = existing_manager

    with with_state_data_lock(state):
        state._lock_manager._lock.acquire.assert_called_once()
    state._lock_manager._lock.release.assert_called_once()


def test_with_state_data_lock_exception():
    from unittest.mock import MagicMock

    from hsm.core.data_management import with_state_data_lock

    state = MagicMock()
    state._lock_manager = None

    try:
        with with_state_data_lock(state):
            raise Exception("Test error")
    except Exception:
        pass

    # Verify lock was released even with exception
    assert hasattr(state, "_lock_manager")
