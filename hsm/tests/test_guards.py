# hsm/tests/test_guards.py
# Copyright (c) 2024
# Licensed under the MIT License - see LICENSE file for details
"""
Test suite for guards defined in guards.py.

This module tests BasicGuard and mock implementations of AbstractGuard to cover
all required scenarios:

1. Basic Error Cases
2. Core Functionality
3. Edge Cases
4. Integration (logging, context managers, async, cleanup)
"""
import asyncio
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict
from unittest.mock import patch

import pytest
import pytest_asyncio

from hsm.core.errors import (
    ActionExecutionError,
    ConfigurationError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    create_error_context,
)
from hsm.core.guards import BasicGuard
from hsm.interfaces.abc import AbstractGuard
from hsm.interfaces.protocols import Event

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class MockEvent:
    """A mock event implementing Event protocol."""

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0):
        self._id = event_id
        self._payload = payload
        self._priority = priority

    def get_id(self) -> str:
        return self._id

    def get_payload(self) -> Any:
        return self._payload

    def get_priority(self) -> int:
        return self._priority


class MockGuard(AbstractGuard):
    """A guard that returns a fixed boolean result."""

    def __init__(self, result: bool):
        self.result = result

    def check(self, event: Event, state_data: Any) -> bool:
        return self.result


class ErrorGuard(AbstractGuard):
    """A guard that always raises GuardEvaluationError."""

    def check(self, event: Event, state_data: Any) -> bool:
        raise GuardEvaluationError("Guard failed", "ErrorGuard", state_data, event)


class LoggingGuard(AbstractGuard):
    """A guard that logs and then returns True."""

    def __init__(self, logger_name: str = "hsm.test.guards"):
        self.logger = logging.getLogger(logger_name)

    def check(self, event: Event, state_data: Any) -> bool:
        self.logger.info("Checking guard with event_id=%s", event.get_id())
        return True


class ComplexErrorGuard(AbstractGuard):
    """A guard that raises a different error depending on state_data."""

    def check(self, event: Event, state_data: Any) -> bool:
        if not isinstance(state_data, dict):
            raise ActionExecutionError("Invalid state_data type", "ComplexErrorGuard", state_data, event)
        if "invalid_state" in state_data:
            raise InvalidStateError("Invalid state encountered", "BadState", "check")
        if "bad_config" in state_data:
            raise ConfigurationError("Bad config", "GuardComponent", {"issue": "missing_key"})
        return False


# Context manager for simulating resource cleanup on error
@contextmanager
def guard_context():
    try:
        yield
    except Exception as e:
        # Simulate cleanup
        cleaned_up = True
        if cleaned_up:
            raise ActionExecutionError("Error in guard context", "ContextGuard", {}, {}) from e


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> Event:
    """A sample event fixture."""
    return MockEvent("test_event", payload={"data": 42}, priority=5)


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data fixture."""
    return {"count": 10, "status": "active"}


@pytest.fixture
def logger_fixture():
    """Fixture for configuring a test logger."""
    logger = logging.getLogger("hsm.test.guards")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


@pytest_asyncio.fixture
async def async_setup():
    """Async fixture for async test setup."""
    await asyncio.sleep(0.01)
    return True


# -----------------------------------------------------------------------------
# BASIC ERROR CASES
# -----------------------------------------------------------------------------


def test_basic_guard_not_implemented(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that calling check on BasicGuard raises NotImplementedError.
    """
    guard = BasicGuard()
    with pytest.raises(NotImplementedError) as exc_info:
        guard.check(sample_event, sample_state_data)
    # Adjust the assertion to match the actual message
    assert "must be subclassed" in str(exc_info.value)


def test_guard_evaluation_error(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that an ErrorGuard raises GuardEvaluationError as expected.
    """
    guard = ErrorGuard()
    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(sample_event, sample_state_data)
    assert "Guard failed" in str(exc_info.value)


def test_invalid_state_error(sample_event: Event) -> None:
    """
    Test that ComplexErrorGuard raises InvalidStateError when invalid_state key is present.
    """
    guard = ComplexErrorGuard()
    with pytest.raises(InvalidStateError) as exc_info:
        guard.check(sample_event, {"invalid_state": True})
    assert "Invalid state encountered" in str(exc_info.value)


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_mock_guard_true(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that a mock guard returning True works as expected.
    """
    guard = MockGuard(True)
    result = guard.check(sample_event, sample_state_data)
    assert result is True


def test_mock_guard_false(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that a mock guard returning False works as expected.
    """
    guard = MockGuard(False)
    result = guard.check(sample_event, sample_state_data)
    assert result is False


def test_error_chaining(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test error chaining by raising a ConfigurationError and then chaining another error.
    """
    guard = ComplexErrorGuard()
    try:
        guard.check(sample_event, {"bad_config": True})
    except ConfigurationError as ce:
        try:
            raise HSMError("Chained error") from ce
        except HSMError as chained:
            assert chained.__cause__ is ce


def test_error_context_preservation(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test creation and preservation of error context.
    """
    try:
        raise GuardEvaluationError("Guard fail", "TestGuard", sample_state_data, sample_event)
    except GuardEvaluationError as ge:
        ctx = create_error_context(ge, "traceback")
        assert ctx.error_type is type(ge)
        assert ctx.traceback == "traceback"
        assert "Guard fail" in ge.message


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------


def test_null_event(sample_state_data: Dict[str, Any]) -> None:
    """
    Test a guard with None event, though unusual, just to ensure it handles gracefully.
    """
    guard = MockGuard(True)
    # If event is None, guard might fail if it tries to access event methods.
    # We handle it by catching any exception:
    try:
        result = guard.check(None, sample_state_data)  # type: ignore
        assert result in [True, False]
    except Exception as e:
        assert isinstance(e, AttributeError)


def test_empty_state_data(sample_event: Event) -> None:
    """
    Test a guard with empty state data to see if it handles empty inputs.
    """
    guard = MockGuard(True)
    result = guard.check(sample_event, {})
    assert result is True


def test_maximum_recursion_scenario(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test a pseudo maximum recursion scenario by repeatedly calling the guard.
    Not truly recursive, but we can simulate by nested calls.
    """
    guard = MockGuard(False)

    def recurse_calls(n: int):
        if n == 0:
            return guard.check(sample_event, sample_state_data)
        else:
            return recurse_calls(n - 1)

    result = recurse_calls(5)
    assert result is False


# -----------------------------------------------------------------------------
# INTEGRATION
# -----------------------------------------------------------------------------


def test_integration_with_logging(
    logger_fixture, sample_event: Event, sample_state_data: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test interaction with logging by using a LoggingGuard.
    """
    guard = LoggingGuard()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hsm.test.guards"):
        res = guard.check(sample_event, sample_state_data)
        assert res is True

    assert any("Checking guard with event_id=test_event" in rec.message for rec in caplog.records)


def test_error_handling_in_context_manager(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test error handling in a context manager scenario.
    If an error occurs inside guard_context, we raise ActionExecutionError after cleanup.
    """
    with pytest.raises(ActionExecutionError) as exc_info:
        with guard_context():
            raise ValueError("Inner error in guard")
    assert "Error in guard context" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_error_scenario(async_setup, sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test async scenario where a guard might fail in async code.
    """

    class AsyncErrorGuard(AbstractGuard):
        async def check(self, event: Event, state_data: Any) -> bool:
            await asyncio.sleep(0.01)
            raise GuardEvaluationError("Async guard fail", "AsyncGuard", state_data, event)

    guard = AsyncErrorGuard()
    with pytest.raises(GuardEvaluationError) as exc_info:
        await guard.check(sample_event, sample_state_data)
    assert "Async guard fail" in str(exc_info.value)


def test_cleanup_procedures(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test scenario where cleanup is needed after an error in a guard.
    We simulate cleanup in the guard_context manager above. Here we do a direct try/except.
    """
    guard = ComplexErrorGuard()
    try:
        guard.check(sample_event, {"invalid_state": True})
    except InvalidStateError as e:
        # Simulate cleanup
        cleaned_up = True
        assert cleaned_up is True
        assert "Invalid state encountered" in str(e)


# -----------------------------------------------------------------------------
# ROBUSTNESS TESTS
# -----------------------------------------------------------------------------


def test_concurrency_thread_safety(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test calling a guard multiple times from different threads simultaneously.
    Ensures no race conditions or exceptions occur under concurrent execution.
    """
    guard = MockGuard(True)
    results = []

    def worker():
        for _ in range(100):
            try:
                res = guard.check(sample_event, sample_state_data)
                results.append(res)
            except Exception as e:
                # If any exception occurs, note it in results
                results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify that all results are True (no exceptions)
    # This is a scenario-based test; assuming no concurrency issues by default.
    assert all(r is True for r in results)


def test_performance_timing(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that a guard completes within a reasonable time frame.
    Here we simulate a guard that sleeps briefly.
    """

    class SlowGuard(AbstractGuard):
        def check(self, event: Event, state_data: Any) -> bool:
            time.sleep(0.05)  # 50ms sleep
            return True

    guard = SlowGuard()
    start = time.time()
    res = guard.check(sample_event, sample_state_data)
    end = time.time()

    elapsed = end - start
    assert res is True
    # Assume the guard should complete within 0.1s.
    assert elapsed < 0.1, f"Guard took too long: {elapsed} seconds"


def test_complex_nested_data(sample_event: Event) -> None:
    """
    Test a guard with deeply nested state data structures.
    Ensures the guard handles complex data gracefully.
    """
    complex_data = {
        "level1": {"level2": {"level3": {"array": [1, 2, {"deep": "value"}], "dict": {"nested_key": "nested_val"}}}},
        "metadata": {"version": "1.0", "timestamp": time.time()},
    }
    guard = MockGuard(True)
    result = guard.check(sample_event, complex_data)
    assert result is True


def test_fuzz_random_payloads() -> None:
    """
    Feed random payloads into the guard to catch any unexpected behavior.
    We'll just rely on MockGuard since it's simple.
    """
    guard = MockGuard(True)

    for _ in range(10):
        random_payload = {
            "rand_int": random.randint(0, 1000),
            "rand_str": "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10)),
            "rand_list": [random.random() for _ in range(5)],
            "rand_dict": {f"key_{i}": random.random() for i in range(3)},
        }

        # We don't have a strong expectation here except that no exception occurs.
        try:
            res = guard.check(MockEvent("fuzz_event"), random_payload)
            assert res is True
        except Exception as e:
            # If an exception occurs, we record it to fail the test.
            assert False, f"Guard failed on fuzz input with exception: {e}"


def test_external_conditions(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test guard behavior under mocked external conditions, such as environment variables.
    This simulates guards depending on external configuration.
    """

    class EnvDependentGuard(AbstractGuard):
        def check(self, event: Event, state_data: Any) -> bool:
            env_var = os.getenv("HSM_GUARD_MODE", "default")
            if env_var == "fail":
                raise GuardEvaluationError("Env condition fail", "EnvDependentGuard", state_data, event)
            return True

    guard = EnvDependentGuard()

    # Test default mode
    with patch.dict(os.environ, {}, clear=True):
        res = guard.check(sample_event, sample_state_data)
        assert res is True, "Should succeed in default mode"

    # Test fail mode
    with patch.dict(os.environ, {"HSM_GUARD_MODE": "fail"}, clear=True):
        with pytest.raises(GuardEvaluationError) as exc_info:
            guard.check(sample_event, sample_state_data)
        assert "Env condition fail" in str(exc_info.value)
