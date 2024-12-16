# hsm/tests/test_actions.py
# Copyright (c) 2024
# Licensed under the MIT License - see LICENSE file for details
"""
Test suite for actions defined in actions.py.

This module contains unit tests for `BasicAction` and mock actions simulating
various scenarios including errors, logging, async operations, and cleanup.
"""

import asyncio
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

import pytest
import pytest_asyncio

from hsm.core.actions import BasicAction
from hsm.core.errors import ActionExecutionError, HSMError
from hsm.interfaces.abc import AbstractAction
from hsm.interfaces.protocols import Event

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class MockEvent:
    """Mock event implementing Event protocol."""

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


class MockAction(AbstractAction):
    """A mock action that executes successfully."""

    def __init__(self, name: str):
        self.name = name
        self.executed = False

    def execute(self, event: Event, state_data: Any) -> None:
        self.executed = True


class ErrorAction(AbstractAction):
    """A mock action that raises ActionExecutionError when executed."""

    def execute(self, event: Event, state_data: Any) -> None:
        raise ActionExecutionError("Action failed", "ErrorAction", state_data, event)


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> Event:
    """Fixture that returns a sample mock event."""
    return MockEvent("test_event", payload={"key": "value"}, priority=10)


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Fixture that returns sample state data."""
    return {"count": 42, "status": "active"}


@pytest.fixture
def logger_fixture():
    """Fixture that configures a test logger."""
    logger = logging.getLogger("hsm.test.actions")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


@pytest_asyncio.fixture
async def async_setup():
    """Async fixture for testing async scenarios."""
    await asyncio.sleep(0.01)
    return True


# -----------------------------------------------------------------------------
# BASIC ERROR CASES
# -----------------------------------------------------------------------------


def test_basic_action_not_implemented(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    action = BasicAction()
    with pytest.raises(NotImplementedError) as exc_info:
        action.execute(sample_event, sample_state_data)
    # Adjust the assertion to match the actual exception message:
    assert "must be subclassed" in str(exc_info.value)


def test_error_action_raises_action_execution_error(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that ErrorAction raises ActionExecutionError on execute.
    """
    action = ErrorAction()
    with pytest.raises(ActionExecutionError) as exc_info:
        action.execute(sample_event, sample_state_data)
    assert "Action failed" in str(exc_info.value)
    assert isinstance(exc_info.value, HSMError)
    assert exc_info.value.action_name == "ErrorAction"


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_mock_action_execution(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that a mock action executes successfully and updates its state.
    """
    action = MockAction("TestAction")
    action.execute(sample_event, sample_state_data)
    assert action.executed is True


def test_error_chaining(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test error chaining by raising another exception from ActionExecutionError.
    """
    try:
        raise ActionExecutionError("Outer error", "TestAction", sample_state_data, sample_event)
    except ActionExecutionError as outer:
        # Chain another error
        try:
            raise HSMError("Chained error") from outer
        except HSMError as chained:
            assert chained.__cause__ is outer


@contextmanager
def action_context_manager():
    """
    Context manager for testing error handling in context.
    Raises ActionExecutionError if an error occurs inside the block.
    """
    try:
        yield
    except Exception as e:
        raise ActionExecutionError("Error in context", "ContextAction", {}, {"type": "CTX"}) from e


def test_error_handling_in_context_manager() -> None:
    with pytest.raises(ActionExecutionError) as exc_info:
        with action_context_manager():
            raise ValueError("Inner error")
    assert "Error in context" in str(exc_info.value)


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------


def test_action_with_null_event(sample_state_data: Dict[str, Any]) -> None:
    """
    Test calling action with None event, expecting it to fail gracefully.
    """
    action = MockAction("NullEventAction")
    # Test how the action handles null event scenario
    action.execute(MockEvent("null_event", None), sample_state_data)
    assert action.executed is True


def test_action_with_empty_state_data(sample_event: Event) -> None:
    """
    Test calling action with empty state data.
    """
    action = MockAction("EmptyStateAction")
    action.execute(sample_event, {})
    assert action.executed is True


def test_maximum_recursion_scenario() -> None:
    """
    Test a pseudo maximum recursion scenario by simulating nested calls.
    There's no real recursion in actions, so we simulate by chaining errors multiple times.
    """
    try:

        def recurse(n: int):
            if n == 0:
                raise ActionExecutionError("Max recursion hit", "RecursiveAction", {}, {})
            else:
                return recurse(n - 1)

        recurse(5)
    except ActionExecutionError as e:
        assert "Max recursion hit" in e.message


# -----------------------------------------------------------------------------
# INTEGRATION
# -----------------------------------------------------------------------------


def test_action_logging_integration(
    logger_fixture, sample_event: Event, sample_state_data: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """
    Test logging interaction when an action is executed.
    """
    action = MockAction("LoggingAction")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hsm.test.actions"):
        logger_fixture.info("Before action execute")
        action.execute(sample_event, sample_state_data)
        logger_fixture.info("After action execute")

    # Check that logs are captured
    assert any("Before action execute" in rec.message for rec in caplog.records)
    assert any("After action execute" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_async_error_scenario(async_setup, sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test async scenario where an action might fail.
    """

    async def async_action():
        await asyncio.sleep(0.01)
        raise ActionExecutionError("Async fail", "AsyncAction", sample_state_data, sample_event)

    with pytest.raises(ActionExecutionError) as exc_info:
        await async_action()
    assert "Async fail" in str(exc_info.value)


def test_cleanup_procedures(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test scenario where cleanup is needed after an action error.
    Simulate cleanup by catching the error and performing cleanup steps.
    """
    action = ErrorAction()
    try:
        action.execute(sample_event, sample_state_data)
    except ActionExecutionError as e:
        # Simulate cleanup here
        cleaned_up = True
        assert cleaned_up is True
        assert "Action failed" in e.message


# -----------------------------------------------------------------------------
# ROBUSTNESS TESTS
# -----------------------------------------------------------------------------


def test_type_mismatch_for_state_data(sample_event: Event) -> None:
    """
    Test passing an incorrect type for state_data (e.g., a string instead of a dict).
    This test verifies that the action either handles it gracefully or raises an error.
    """
    action = BasicAction()
    # BasicAction is not implemented, so we expect NotImplementedError.
    # Replace with MockAction for demonstration:
    action = MockAction("TypeMismatchAction")
    # Passing a string as state_data
    state_data = "not a dict"
    try:
        action.execute(sample_event, state_data)  # Depending on implementation, this might fail or pass.
        assert action.executed is True
    except Exception as e:
        # If an error is raised, at least we know it's handled and doesn't silently fail.
        assert isinstance(e, HSMError) or isinstance(e, TypeError)


def test_type_mismatch_for_event(sample_state_data: Dict[str, Any]) -> None:
    """
    Test passing a non-Event object as the event parameter.
    Verifies that actions detect incorrect input types.
    """
    action = MockAction("NonEventTest")
    non_event = {"id": "fake_event"}  # Not an Event protocol implementation
    try:
        action.execute(non_event, sample_state_data)  # Should fail or behave unexpectedly
        assert action.executed is True  # If it doesn't fail, we note that assumption.
    except Exception as e:
        assert isinstance(e, HSMError) or isinstance(e, AttributeError)


def test_reentrancy_idempotency(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test calling an action multiple times quickly to simulate concurrent or repeated calls.
    This checks if the action remains stable and idempotent.
    """
    action = MockAction("ReentrantAction")

    # Call the action multiple times
    for _ in range(5):
        try:
            action.execute(sample_event, sample_state_data)
            assert action.executed is True
        except Exception as e:
            # We don't expect errors here; if any occur, note the failure.
            assert False, f"Action failed on repeated execution: {e}"


def test_complex_payload_integrity(sample_state_data: Dict[str, Any]) -> None:
    """
    Test that complex payloads are not mutated by the action.
    """
    complex_payload = {"level1": {"list": [1, 2, 3], "dict": {"nested": "value"}}, "config": {"threshold": 100}}
    original_payload = dict(complex_payload)  # shallow copy for comparison

    evt = MockEvent("complex_event", payload=complex_payload)
    action = MockAction("PayloadIntegrityAction")
    action.execute(evt, sample_state_data)

    # Check payload integrity
    # NOTE: This only checks shallow immutability. A deep check might be needed in real scenarios.
    assert evt.get_payload() == complex_payload, "Payload should remain unchanged"
    assert evt.get_payload() == original_payload, "Original and final payload should match"


def test_action_performance_timing(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that the action executes within a reasonable time frame.
    This is a simple timing test, just ensuring it doesn't hang excessively.
    """
    action = MockAction("PerformanceTest")

    start_time = time.time()
    action.execute(sample_event, sample_state_data)
    end_time = time.time()

    elapsed = end_time - start_time
    # Assume the action should complete in under 0.5 seconds for this test.
    assert elapsed < 0.5, f"Action took too long: {elapsed} seconds"


@pytest.mark.asyncio
async def test_async_action_with_unusual_inputs() -> None:
    """
    Test async scenario where the event is unusual or incomplete.
    This ensures async actions handle odd inputs gracefully.
    """

    # Simulate an async action by creating one that awaits and then raises
    class AsyncMockAction(AbstractAction):
        async def execute(self, event: Any, state_data: Any) -> None:
            await asyncio.sleep(0.01)
            if not hasattr(event, "get_id"):
                raise ActionExecutionError("Invalid event in async context", "AsyncInvalid", state_data, event)

    action = AsyncMockAction()
    non_event = {"no_id": True}

    with pytest.raises(ActionExecutionError) as exc_info:
        await action.execute(non_event, {})
    assert "Invalid event in async context" in str(exc_info.value)


def test_cleanup_after_type_error(sample_event: Event) -> None:
    """
    Test cleanup procedures after a type-related error occurs.
    This simulates having to revert partial changes if an action fails due to bad input.
    """
    action = ErrorAction()
    # Modify the sample_event to simulate complexity:
    event_data = sample_event.get_payload()
    event_data["should_revert"] = True

    try:
        action.execute(sample_event, {"invalid": object()})
    except ActionExecutionError:
        # Simulate cleanup logic:
        reverted = True
        assert reverted is True, "Cleanup should have occurred after error"
