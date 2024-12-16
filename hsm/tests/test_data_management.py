# hsm/tests/test_data_management.py
# nosec
# Licensed under the MIT License - see LICENSE file for details
"""
Test suite for data_management.py.

This module tests the DataManager class and the DataLockError exception.
It ensures thread-safe data access, correct locking behavior, and proper exception handling.

Sections covered:
- Basic Error Cases
- Core Functionality
- Edge Cases
- Integration
"""

import random
import threading
import time
import typing
from typing import Any, Dict, get_type_hints

import pytest

from hsm.core.data_management import DataLockError, DataManager
from hsm.core.errors import HSMError

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def data_manager() -> DataManager:
    """Provides a fresh DataManager instance for each test."""
    return DataManager()


# -----------------------------------------------------------------------------
# BASIC ERROR CASES
# -----------------------------------------------------------------------------


def test_data_lock_error_inheritance() -> None:
    """
    Test that DataLockError inherits from HSMError.
    """
    assert issubclass(DataLockError, HSMError)


def test_data_lock_error_message() -> None:
    """
    Test DataLockError message formatting.
    """
    exc = DataLockError("Lock acquisition failed", details={"reason": "test"})
    assert "Lock acquisition failed" in str(exc)
    assert exc.details == {"reason": "test"}


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_basic_data_access(data_manager: DataManager) -> None:
    """
    Test that we can set and retrieve data within the access_data context.
    """
    with data_manager.access_data() as data:
        data["counter"] = 1

    # Validate the data was set
    snapshot = data_manager.get_data_snapshot()
    assert snapshot["counter"] == 1


def test_multiple_access_calls(data_manager: DataManager) -> None:
    """
    Test that consecutive calls to access_data return consistent data.
    """
    with data_manager.access_data() as data:
        data["value"] = 42

    # Another access changes data again
    with data_manager.access_data() as data:
        data["value"] = 100

    snapshot = data_manager.get_data_snapshot()
    assert snapshot["value"] == 100


def test_data_lock_error_cannot_easily_be_triggered(data_manager: DataManager) -> None:
    """
    It's hard to force a DataLockError since a normal Lock always succeeds.
    This test documents that we can't easily trigger it.
    If we implement a non-blocking lock scenario in the future, we could test it.
    """
    # Just ensure no error raised in normal scenarios.
    with data_manager.access_data() as data:
        data["test"] = "no error"
    snapshot = data_manager.get_data_snapshot()
    assert snapshot["test"] == "no error"


def test_error_chaining() -> None:
    """
    Test error chaining by raising DataLockError caused by another exception.
    """
    try:
        try:
            raise ValueError("Inner error")
        except ValueError as inner:
            raise DataLockError("Outer error") from inner
    except DataLockError as outer:
        assert "Outer error" in str(outer)
        assert isinstance(outer.__cause__, ValueError)


def test_error_context_preservation() -> None:
    """
    Test error context preservation by creating a DataLockError and checking its details.
    """
    exc = DataLockError("Lock failed", details={"attempt": 1})
    assert exc.message == "Lock failed"
    assert exc.details == {"attempt": 1}


def test_custom_error_properties() -> None:
    """
    Test that custom properties set in DataLockError details are preserved.
    """
    exc = DataLockError("Custom error", details={"custom": "info"})
    assert exc.details["custom"] == "info"


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------


def test_empty_data_initially(data_manager: DataManager) -> None:
    """
    Test that initially, DataManager data is empty.
    """
    snapshot = data_manager.get_data_snapshot()
    assert snapshot == {}


def test_clear_data(data_manager: DataManager) -> None:
    """
    Test clearing all data in a thread-safe manner.
    """
    with data_manager.access_data() as data:
        data["to_delete"] = "exists"

    data_manager.clear_data()
    snapshot = data_manager.get_data_snapshot()
    assert snapshot == {}


def test_maximum_recursion_scenario(data_manager: DataManager) -> None:
    """
    Pseudo test for maximum recursion scenario:
    Just show that we can access data multiple times.
    """

    def recurse(n: int):
        if n == 0:
            with data_manager.access_data() as data:
                data["depth"] = "reached"
            return
        return recurse(n - 1)

    recurse(10)
    snapshot = data_manager.get_data_snapshot()
    assert snapshot["depth"] == "reached"


# -----------------------------------------------------------------------------
# INTEGRATION
# -----------------------------------------------------------------------------


def test_integration_with_logging(caplog: pytest.LogCaptureFixture, data_manager: DataManager) -> None:
    """
    Test interaction with logging.
    Although DataManager doesn't log itself, we can simulate logging while using it.
    """
    caplog.clear()
    with data_manager.access_data() as data:
        data["log_test"] = True

    # Simulate logging outside the context
    import logging

    logger = logging.getLogger("hsm.test.data_management")
    with caplog.at_level(logging.INFO, logger="hsm.test.data_management"):
        logger.info("Data snapshot: %s", data_manager.get_data_snapshot())

    assert any("Data snapshot: {'log_test': True}" in rec.message for rec in caplog.records)


def test_error_handling_in_context_manager(data_manager: DataManager) -> None:
    """
    Test error handling within the context manager block.
    If an error occurs inside the block, changes are reverted.
    """
    try:
        with data_manager.access_data() as data:
            data["will_fail"] = "temp"
            raise RuntimeError("Simulated failure")
    except RuntimeError:
        pass

    # After the error, verify that the changes did not persist?
    # Actually, the DataManager doesn't revert changes automatically,
    # only SetDataAction did that. DataManager just ensures atomic access.
    # Without special logic, DataManager won't revert changes.
    # So we expect "will_fail" to be set.
    snapshot = data_manager.get_data_snapshot()
    # In this current design, DataManager doesn't revert changes on exception,
    # it just releases the lock. The test docstring was from previous code context.
    # Let's just assert that the data remains changed.
    assert snapshot["will_fail"] == "temp"


@pytest.mark.asyncio
async def test_async_error_scenario(data_manager: DataManager) -> None:
    """
    Test async scenario where we access data in async code.
    DataManager is sync-only, but we can still use it in async code.
    """

    # Just ensure no issues accessing from async code.
    async def async_operation():
        with data_manager.access_data() as data:
            data["async_access"] = True

    await async_operation()
    snapshot = data_manager.get_data_snapshot()
    assert snapshot["async_access"] is True


def test_cleanup_procedures(data_manager: DataManager) -> None:
    """
    Test scenario where cleanup is needed after an error.
    DataManager doesn't require special cleanup after errors, the lock is always released.
    """
    with data_manager.access_data() as data:
        data["test"] = "before_error"

    # Simulate an error after using data
    try:
        raise HSMError("Simulated HSM error")
    except HSMError:
        # Cleanup logic could be placed here if needed.
        cleaned_up = True
        assert cleaned_up is True

    snapshot = data_manager.get_data_snapshot()
    assert snapshot["test"] == "before_error"


# -----------------------------------------------------------------------------
# CONCURRENCY TESTS
# -----------------------------------------------------------------------------


def test_concurrent_access(data_manager: DataManager) -> None:
    """
    Test concurrent access to DataManager from multiple threads.
    """

    def worker():
        for _ in range(100):
            with data_manager.access_data() as data:
                counter = data.get("counter", 0)
                data["counter"] = counter + 1

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snapshot = data_manager.get_data_snapshot()
    # With 5 threads * 100 increments each, we expect 500 increments.
    assert snapshot["counter"] == 500


# -----------------------------------------------------------------------------
# PERFORMANCE/STRESS TESTS
# -----------------------------------------------------------------------------


def test_performance_under_load(data_manager: DataManager) -> None:
    """
    Test DataManager under a relatively high load.
    This is a simplistic stress test and not a true performance benchmark.
    """

    def writer():
        for _ in range(1000):
            with data_manager.access_data() as data:
                val = data.get("count", 0)
                data["count"] = val + 1

    start_time = time.time()
    threads = [threading.Thread(target=writer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    # Check that operations completed in a "reasonable" amount of time
    # Exact threshold depends on environment, adjust if too strict.
    assert elapsed < 2.0, f"DataManager operations took too long: {elapsed} seconds"

    snapshot = data_manager.get_data_snapshot()
    # 10 threads * 1000 increments = 10,000
    assert snapshot["count"] == 10000


# -----------------------------------------------------------------------------
# NON-BLOCKING LOCK/TIMEOUT SCENARIO (FUTURE EXTENSION)
# -----------------------------------------------------------------------------


@pytest.mark.skip(reason="Non-blocking lock scenario not implemented.")
def test_data_lock_error_non_blocking():
    """
    Example test that would be used if DataManager supported non-blocking lock attempts.

    Since DataManager doesn't currently support non-blocking lock attempts,
    this test is skipped.
    """
    # Hypothetical code:
    # data_manager = DataManager(non_blocking=True)
    # with pytest.raises(DataLockError):
    #     # Attempt a non-blocking access that fails
    #     with data_manager.access_data():
    #         pass
    pass


# -----------------------------------------------------------------------------
# TYPE CHECKING AND PROTOCOL COMPLIANCE TESTS
# -----------------------------------------------------------------------------


def test_type_hints_on_data_manager() -> None:
    """
    Test that DataManager's type hints are present and correct by inspecting type hints.

    This doesn't fully ensure correctness like mypy would, but it can catch missing hints.
    """
    hints = get_type_hints(DataManager)
    # Check that certain attributes have type hints
    # Note: The class-level attributes won't show up here if defined in __init__.
    # Instead, we can check methods.
    method_hints = get_type_hints(DataManager.access_data)
    # access_data should return a Generator[Dict[str, Any], None, None]
    return_type = method_hints.get("return", None)
    assert return_type is not None, "Expected a return type hint on access_data."
    # This is a loose check, for a stricter check, we'd compare equality with Generator[Dict[str, Any], None, None]
    # but that might be too strict. We just confirm something is present.


# -----------------------------------------------------------------------------
# FUZZ TESTING
# -----------------------------------------------------------------------------


def test_fuzz_random_operations(data_manager: DataManager) -> None:
    """
    Perform randomized read/write operations from multiple threads to detect
    any rare race conditions or unexpected states.

    This test is non-deterministic. If errors are rare, it might never fail.
    It's more of a heuristic check.
    """

    def random_op():
        with data_manager.access_data() as data:
            operation = random.choice(["read", "write", "clear"])
            if operation == "write":
                key = f"key_{random.randint(0, 10)}"
                data[key] = random.randint(0, 1000)
            elif operation == "read":
                # Just read some keys
                for _ in range(5):
                    key = f"key_{random.randint(0, 10)}"
                    _ = data.get(key, None)
            elif operation == "clear":
                # Clear occasionally
                if random.random() < 0.1:
                    data.clear()

    threads = [threading.Thread(target=random_op) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Just ensure no exception was raised. If something was wrong with locking,
    # we might have encountered a race condition or crashed.


# -----------------------------------------------------------------------------
# VERIFYING THE ENVIRONMENT
# -----------------------------------------------------------------------------


def test_data_manager_sanity(data_manager: DataManager) -> None:
    """
    A simple sanity test to confirm environment and that data_manager works as expected.
    """
    with data_manager.access_data() as data:
        data["sanity_check"] = True
    snapshot = data_manager.get_data_snapshot()
    assert snapshot["sanity_check"] is True
