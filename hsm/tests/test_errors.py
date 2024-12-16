# hsm/tests/test_errors.py
# nosec
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
"""Test suite for HSM error handling and error classes.

This module contains unit tests for various error conditions and error
classes in the HSM implementation.
"""
import dataclasses
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator

import pytest
import pytest_asyncio

from hsm.core.errors import (
    ActionExecutionError,
    ConcurrencyError,
    ConfigurationError,
    ErrorContext,
    EventQueueFullError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    InvalidTransitionError,
    create_error_context,
)


# Fixtures
@pytest.fixture
def sample_event() -> Dict[str, Any]:
    """Sample event data for testing."""
    return {"type": "TEST_EVENT", "data": {"value": 42}}


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data for testing."""
    return {"counter": 0, "status": "ready"}


@pytest.fixture
def error_details() -> Dict[str, Any]:
    """Sample error details for testing."""
    return {"timestamp": time.time(), "severity": "high"}


@pytest.fixture
def complex_state_data() -> Dict[str, Any]:
    """Complex nested state data for thorough testing."""
    return {
        "level1": {"level2": {"value": 42, "status": "active"}, "array": [1, 2, 3]},
        "metadata": {"version": "1.0", "timestamp": time.time()},
    }


@pytest.fixture
def mock_logger():
    """Fixture for testing error logging."""
    logger = logging.getLogger("hsm.test")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


# -----------------------------------------------------------------------------
# BASIC ERROR CASES TESTS
# -----------------------------------------------------------------------------
def test_hsm_error_base():
    """Test the base HSM error class."""
    message = "Test error message"
    details = {"key": "value"}
    error = HSMError(message, details)

    assert str(error) == message
    assert error.message == message
    assert error.details == details


def test_invalid_transition_error():
    """Test InvalidTransitionError creation and attributes."""
    source = "STATE_A"
    target = "STATE_B"
    event = {"type": "TRANSITION"}
    details = {"reason": "guard failed"}

    error = InvalidTransitionError("Invalid transition", source, target, event, details)

    assert error.source_state == source
    assert error.target_state == target
    assert error.event == event
    assert error.details == details


def test_invalid_state_error():
    """Test InvalidStateError creation and attributes."""
    state_id = "INVALID_STATE"
    operation = "entry_action"
    error = InvalidStateError("Invalid state", state_id, operation)

    assert error.state_id == state_id
    assert error.operation == operation


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY TESTS
# -----------------------------------------------------------------------------
def test_configuration_error():
    """Test ConfigurationError with validation errors."""
    component = "StateMachine"
    validation_errors = {"missing_states": ["STATE_A", "STATE_B"], "invalid_transitions": [("STATE_A", "STATE_C")]}

    error = ConfigurationError("Configuration invalid", component, validation_errors)

    assert error.component == component
    assert error.validation_errors == validation_errors


def test_guard_evaluation_error(sample_event, sample_state_data):
    """Test GuardEvaluationError with state data and event."""
    guard_name = "check_condition"
    error = GuardEvaluationError("Guard evaluation failed", guard_name, sample_state_data, sample_event)

    assert error.guard_name == guard_name
    assert error.state_data == sample_state_data
    assert error.event == sample_event


# -----------------------------------------------------------------------------
# CONTEXT MANAGER TESTS
# -----------------------------------------------------------------------------
@contextmanager
def error_context() -> Generator[None, None, None]:
    """Context manager for testing error handling."""
    try:
        yield
    except HSMError as e:
        raise e


def test_error_context_manager():
    """Test errors within context manager."""
    with pytest.raises(InvalidStateError):
        with error_context():
            raise InvalidStateError("Context test", "STATE_A", "test_op")


# -----------------------------------------------------------------------------
# EDGE CASES TESTS
# -----------------------------------------------------------------------------
def test_error_with_empty_details():
    """Test error creation with empty details."""
    error = HSMError("Test message")
    assert error.details == {}


def test_error_context_creation():
    """Test ErrorContext creation and attributes."""
    error = HSMError("Test error")
    traceback = "Traceback (most recent call last):\n..."
    context = create_error_context(error, traceback)

    assert isinstance(context, ErrorContext)
    assert context.error_type == HSMError
    assert context.traceback == traceback
    assert isinstance(context.timestamp, float)


def test_error_context_immutability():
    """Test that ErrorContext is immutable once created."""
    error = HSMError("Test error")
    context = create_error_context(error, "traceback")

    with pytest.raises(dataclasses.FrozenInstanceError):
        context.timestamp = time.time()


def test_nested_error_handling():
    """Test nested error scenarios."""
    try:
        try:
            raise InvalidStateError("Inner error", "STATE_A", "op")
        except InvalidStateError as inner:
            raise ConfigurationError("Outer error", "component", details={"inner_error": str(inner)})
    except ConfigurationError as outer:
        assert "inner_error" in outer.details


# -----------------------------------------------------------------------------
# INTEGRATION TESTS
# -----------------------------------------------------------------------------
def test_event_queue_full_error():
    """Test EventQueueFullError in queue context."""
    queue_size = 100
    max_size = 100
    event = {"type": "TEST_EVENT"}

    error = EventQueueFullError("Queue is full", queue_size, max_size, event)

    assert error.queue_size == queue_size
    assert error.max_size == max_size
    assert error.dropped_event == event


def test_concurrency_error():
    """Test ConcurrencyError in concurrent operations."""
    operation = "state_transition"
    resource = "state_data"

    error = ConcurrencyError("Concurrent access denied", operation, resource)

    assert error.operation == operation
    assert error.resource == resource


def test_action_execution_error(sample_event, sample_state_data):
    """Test ActionExecutionError during action execution."""
    action_name = "update_state"
    error = ActionExecutionError("Action execution failed", action_name, sample_state_data, sample_event)

    assert error.action_name == action_name
    assert error.state_data == sample_state_data
    assert error.event == sample_event


@pytest_asyncio.fixture
async def async_fixture():
    """Fixture for async tests."""
    return True


@pytest.mark.asyncio
async def test_async_error_handling(async_fixture):
    """Test error handling in async context."""

    async def async_operation():
        raise InvalidStateError("Async error", "STATE_A", "async_op")

    with pytest.raises(InvalidStateError) as exc_info:
        await async_operation()

    assert exc_info.value.state_id == "STATE_A"
    assert exc_info.value.operation == "async_op"


def test_error_with_complex_data(complex_state_data):
    """Test error handling with complex nested data structures."""
    error = GuardEvaluationError("Complex data error", "complex_guard", complex_state_data, {"type": "COMPLEX_EVENT"})
    assert error.state_data == complex_state_data


def test_error_logging(mock_logger):
    """Test error logging functionality."""
    error = HSMError("Logged error", {"log_level": "ERROR"})
    mock_logger.error(str(error), extra=error.details)
    # Verify logging occurred correctly - would need proper assertions based on logging setup


def test_error_chaining():
    """Test error chaining capabilities."""
    try:
        try:
            raise InvalidStateError("Original error", "STATE_A", "op")
        except InvalidStateError as e:
            raise ActionExecutionError("Chained error", "action", {}, {}) from e
    except ActionExecutionError as e:
        assert isinstance(e.__cause__, InvalidStateError)


def test_error_with_none_values():
    """Test error handling with None values."""
    error = HSMError("Test with None", {"null_value": None})
    assert error.details["null_value"] is None


def test_error_inheritance():
    """Test error class inheritance hierarchy."""
    errors = [
        InvalidTransitionError("test", "A", "B", {}),
        InvalidStateError("test", "A", "op"),
        ConfigurationError("test", "comp"),
        GuardEvaluationError("test", "guard", {}, {}),
        ActionExecutionError("test", "action", {}, {}),
        ConcurrencyError("test", "op", "res"),
        EventQueueFullError("test", 10, 10, {}),
    ]

    for error in errors:
        assert isinstance(error, HSMError)


def test_error_context_properties():
    """Test that ErrorContext properties are correctly set and accessible."""
    error = HSMError("Test error")
    context = create_error_context(error, "traceback")

    # Verify properties are set correctly
    assert isinstance(context.timestamp, float)
    assert context.error_type == HSMError
    assert context.traceback == "traceback"
    assert isinstance(context.details, dict)


def test_error_details_modification():
    """Test modification of error details."""
    error = HSMError("Test error", {"original": "value"})
    error.details["new"] = "value"
    assert "new" in error.details
    assert error.details["original"] == "value"
