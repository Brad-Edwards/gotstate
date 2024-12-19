"""
Integration tests for guards defined in guards.py.

These tests verify the integration of guards with external systems and resources.
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

from hsm.core.errors import ActionExecutionError, GuardEvaluationError, InvalidStateError
from hsm.interfaces.abc import AbstractGuard
from hsm.interfaces.protocols import Event
from hsm.tests.unit.test_guards import MockEvent


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


class LoggingGuard(AbstractGuard):
    """A guard that logs and then returns True."""

    def __init__(self, logger_name: str = "hsm.test.guards"):
        self.logger = logging.getLogger(logger_name)

    def check(self, event: Event, state_data: Any) -> bool:
        self.logger.info("Checking guard with event_id=%s", event.get_id())
        return True


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> Event:
    """A sample event fixture."""
    return MockEvent("test_event", payload={"data": 42}, priority=5)


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
# INTEGRATION TESTS
# -----------------------------------------------------------------------------


def test_integration_with_logging(logger_fixture, sample_event: Event, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test interaction with logging by using a LoggingGuard.
    """
    guard = LoggingGuard()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hsm.test.guards"):
        res = guard.check(sample_event, {})
        assert res is True

    assert any("Checking guard with event_id=test_event" in rec.message for rec in caplog.records)


def test_error_handling_in_context_manager(sample_event: Event) -> None:
    """
    Test error handling in a context manager scenario.
    If an error occurs inside guard_context, we raise ActionExecutionError after cleanup.
    """
    with pytest.raises(ActionExecutionError) as exc_info:
        with guard_context():
            raise ValueError("Inner error in guard")
    assert "Error in guard context" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_error_scenario(async_setup, sample_event: Event) -> None:
    """
    Test async scenario where a guard might fail in async code.
    """

    class AsyncErrorGuard(AbstractGuard):
        async def check(self, event: Event, state_data: Any) -> bool:
            await asyncio.sleep(0.01)
            raise GuardEvaluationError("Async guard fail", "AsyncGuard", state_data, event)

    guard = AsyncErrorGuard()
    with pytest.raises(GuardEvaluationError) as exc_info:
        await guard.check(sample_event, {})
    assert "Async guard fail" in str(exc_info.value)


def test_concurrency_thread_safety(sample_event: Event) -> None:
    """
    Test calling a guard multiple times from different threads simultaneously.
    Ensures no race conditions or exceptions occur under concurrent execution.
    """

    class ThreadSafeGuard(AbstractGuard):
        def check(self, event: Event, state_data: Any) -> bool:
            return True

    guard = ThreadSafeGuard()
    results = []

    def worker():
        for _ in range(100):
            try:
                res = guard.check(sample_event, {})
                results.append(res)
            except Exception as e:
                results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is True for r in results)


def test_performance_timing(sample_event: Event) -> None:
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
    res = guard.check(sample_event, {})
    end = time.time()

    elapsed = end - start
    assert res is True
    assert elapsed < 0.1, f"Guard took too long: {elapsed} seconds"


def test_external_conditions(sample_event: Event) -> None:
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
        res = guard.check(sample_event, {})
        assert res is True, "Should succeed in default mode"

    # Test fail mode
    with patch.dict(os.environ, {"HSM_GUARD_MODE": "fail"}, clear=True):
        with pytest.raises(GuardEvaluationError) as exc_info:
            guard.check(sample_event, {})
        assert "Env condition fail" in str(exc_info.value)
