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
