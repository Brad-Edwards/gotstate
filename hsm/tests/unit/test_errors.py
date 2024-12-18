# hsm/tests/test_errors.py
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
    ValidationError,
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


@pytest.fixture
def error_base_data():
    """Common test data for error cases."""
    return {
        "message": "Test error message",
        "details": {"key": "value"},
        "state_id": "STATE_A",
        "operation": "test_op",
        "component": "state_machine",
    }


# -----------------------------------------------------------------------------
# BASIC ERROR CASES TESTS
# -----------------------------------------------------------------------------
def test_hsm_error_base(error_base_data):
    """Test the base HSM error class."""
    error = HSMError(error_base_data.get("message"), error_base_data.get("details"))

    assert str(error) == error_base_data.get("message")
    assert error.message == error_base_data.get("message")
    assert error.details == error_base_data.get("details")


def test_invalid_transition_error(error_base_data):
    """Test InvalidTransitionError creation and attributes."""
    source = error_base_data.get("state_id")
    target = "STATE_B"
    event = {"type": "TRANSITION"}
    details = {"reason": "guard failed"}

    error = InvalidTransitionError("Invalid transition", source, target, event, details)

    assert error.source_state == source
    assert error.target_state == target
    assert error.event == event
    assert error.details == details


def test_invalid_state_error(error_base_data):
    """Test InvalidStateError creation and attributes."""
    error = InvalidStateError(
        error_base_data.get("message"), error_base_data.get("state_id"), error_base_data.get("operation")
    )
    assert error.state_id == error_base_data.get("state_id")
    assert error.operation == error_base_data.get("operation")


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
        # Add useful handling like logging or error transformation
        logging.error(f"HSM error occurred: {str(e)}")
        # Could also add context or transform the error
        e.details["handled_by"] = "error_context"
        raise


def test_error_context_manager():
    """Test errors within context manager."""
    with pytest.raises(InvalidStateError) as exc_info:
        with error_context():
            raise InvalidStateError("Context test", "STATE_A", "test_op")

    assert exc_info.value.details["handled_by"] == "error_context"


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
    verify_error_context(context, HSMError, traceback)


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


# -----------------------------------------------------------------------------
# VALIDATION ERROR TESTS
# -----------------------------------------------------------------------------


def test_validation_error_basic() -> None:
    """Test basic ValidationError creation and attributes."""
    error = ValidationError("Validation failed")
    assert str(error) == "Validation failed"
    assert error.component is None
    assert error.validation_results == []
    assert error.details == {}


def test_validation_error_with_component() -> None:
    """Test ValidationError with component information."""
    error = ValidationError("Validation failed", component="state_machine")
    assert error.component == "state_machine"
    assert error.validation_results == []


def test_validation_error_with_results() -> None:
    """Test ValidationError with validation results."""
    results = [
        {"severity": "ERROR", "message": "Invalid state"},
        {"severity": "WARNING", "message": "Missing transition"},
    ]
    error = ValidationError("Validation failed", component="state_machine", validation_results=results)
    assert error.validation_results == results
    assert len(error.validation_results) == 2


def test_validation_error_with_details() -> None:
    """Test ValidationError with additional details."""
    details = {"timestamp": 123456789, "validator": "structure"}
    error = ValidationError("Validation failed", component="state_machine", details=details)
    assert error.details == details


def test_validation_error_inheritance() -> None:
    """Test that ValidationError properly inherits from HSMError."""
    error = ValidationError("Validation failed")
    assert isinstance(error, HSMError)
    assert isinstance(error, Exception)


# -----------------------------------------------------------------------------
# ERROR MESSAGE FORMATTING TESTS
# -----------------------------------------------------------------------------
def test_error_message_formatting():
    """Test error message formatting with various inputs."""
    # Test with special characters
    error = HSMError("Test: $pecial @#$%^&* chars")
    assert str(error) == "Test: $pecial @#$%^&* chars"

    # Test with multi-line message
    message = "Line 1\nLine 2\nLine 3"
    error = HSMError(message)
    assert str(error) == message

    # Test with empty message
    error = HSMError("")
    assert str(error) == ""


def test_error_message_with_unicode():
    """Test error message handling with Unicode characters."""
    messages = [
        "Unicode: 你好世界",  # Chinese
        "Unicode: Привет мир",  # Russian
        "Unicode: 🌍🌎🌏",  # Emojis
    ]
    for message in messages:
        error = HSMError(message)
        assert str(error) == message


# -----------------------------------------------------------------------------
# ERROR CHAINING TESTS
# -----------------------------------------------------------------------------
def test_error_chaining_depth():
    """Test error chaining with multiple levels."""
    try:
        try:
            try:
                raise ValueError("Root cause")
            except ValueError as e:
                raise InvalidStateError("Level 1", "STATE_A", "op") from e
        except InvalidStateError as e:
            raise GuardEvaluationError("Level 2", "guard", {}, {}) from e
    except GuardEvaluationError as e:
        assert isinstance(e.__cause__, InvalidStateError)
        assert isinstance(e.__cause__.__cause__, ValueError)
        assert str(e.__cause__.__cause__) == "Root cause"


def test_error_chaining_without_cause():
    """Test error chaining behavior without explicit cause."""
    try:
        raise InvalidStateError("No cause error", "STATE_A", "op")
    except InvalidStateError as e:
        assert e.__cause__ is None


# -----------------------------------------------------------------------------
# ERROR CONTEXT SERIALIZATION TESTS
# -----------------------------------------------------------------------------
def test_error_context_serialization():
    """Test ErrorContext serialization with various data types."""
    error = HSMError("Test error")
    context = create_error_context(error, "test traceback")

    # Test serialization of basic attributes
    assert isinstance(context.timestamp, float)
    assert context.error_type == HSMError
    assert context.traceback == "test traceback"

    # Test with complex details
    error_with_details = HSMError(
        "Test error",
        {
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "nested": {"a": {"b": {"c": 1}}},
        },
    )
    context = create_error_context(error_with_details, "test traceback")
    assert isinstance(context.details, dict)
    assert context.details["int"] == 42
    assert context.details["nested"]["a"]["b"]["c"] == 1


def test_error_context_with_large_traceback():
    """Test ErrorContext with large traceback."""
    large_traceback = "Line\n" * 1000
    error = HSMError("Test error")
    context = create_error_context(error, large_traceback)
    assert context.traceback == large_traceback


# -----------------------------------------------------------------------------
# BOUNDARY CONDITION TESTS
# -----------------------------------------------------------------------------
def test_error_with_large_details():
    """Test error handling with large detail structures."""
    # Create a large nested dictionary
    large_dict = {}
    for i in range(100):
        large_dict[f"key_{i}"] = {"data": "x" * 100, "nested": {"value": i}}

    error = HSMError("Large details test", large_dict)
    assert error.details == large_dict


def test_error_with_circular_reference():
    """Test error handling with circular references in details."""
    details = {"key": "value"}
    details["self"] = details  # Create circular reference

    # Should handle circular reference gracefully
    error = HSMError("Circular reference test", details)
    assert error.details["self"] is error.details


def test_error_recovery_scenarios():
    """Test error recovery and cleanup scenarios."""

    class CleanupTracker:
        def __init__(self):
            self.cleaned_up = False

        def cleanup(self):
            self.cleaned_up = True

    tracker = CleanupTracker()

    try:
        try:
            raise InvalidStateError("Test error", "STATE_A", "op")
        except InvalidStateError:
            tracker.cleanup()
            raise
    except InvalidStateError as e:
        assert tracker.cleaned_up
        assert isinstance(e, InvalidStateError)


def test_validation_error_with_empty_results():
    """Test ValidationError with empty validation results."""
    error = ValidationError("Empty validation")
    assert error.validation_results == []
    assert error.component is None


def test_validation_error_with_large_results():
    """Test ValidationError with large validation results."""
    large_results = [{"severity": "ERROR", "message": f"Error {i}", "details": {"data": "x" * 100}} for i in range(100)]
    error = ValidationError("Large validation", validation_results=large_results)
    assert len(error.validation_results) == 100
    assert error.validation_results == large_results


def verify_error_context(context, error_type, expected_traceback):
    """Helper to verify ErrorContext properties."""
    assert isinstance(context, ErrorContext)
    assert isinstance(context.timestamp, float)
    assert context.error_type == error_type
    assert context.traceback == expected_traceback
    assert isinstance(context.details, dict)


@pytest.mark.parametrize(
    "test_data",
    [
        {"message": "Validation failed"},
        {"message": "Validation failed", "component": "state_machine"},
        {
            "message": "Validation failed",
            "component": "state_machine",
            "validation_results": [{"severity": "ERROR", "message": "Invalid state"}],
        },
        {"message": "Validation failed", "details": {"timestamp": 123456789}},
    ],
)
def test_validation_error_variants(test_data):
    """Test ValidationError with various configurations."""
    error = ValidationError(**test_data)
    assert str(error) == test_data.get("message")
    if "component" in test_data:
        assert error.component == test_data.get("component")
    if "validation_results" in test_data:
        assert error.validation_results == test_data.get("validation_results")
    if "details" in test_data:
        assert error.details == test_data.get("details")
