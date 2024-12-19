# hsm/tests/unit/test_errors.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

"""Test suite for error classes defined in errors.py."""

import time
from typing import Any, Dict

import pytest

from hsm.core.errors import (
    ActionExecutionError,
    ConcurrencyError,
    ConfigurationError,
    ErrorContext,
    EventQueueFullError,
    ExecutionError,
    ExecutorError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    InvalidTransitionError,
    StateError,
    ValidationError,
    create_error_context,
)

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event():
    """Sample event for testing."""

    class MockEvent:
        def get_id(self):
            return "test_event"

    return MockEvent()


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data for testing."""
    return {"status": "active", "count": 42}


# -----------------------------------------------------------------------------
# BASE ERROR TESTS
# -----------------------------------------------------------------------------


def test_hsm_error_basic():
    """Test basic HSMError functionality."""
    error = HSMError("test message")
    assert str(error) == "test message"
    assert error.message == "test message"
    assert error.details == {}

    # Test with details
    details = {"key": "value"}
    error = HSMError("test message", details)
    assert error.details == details


def test_executor_error():
    """Test ExecutorError functionality."""
    error = ExecutorError("executor failed")
    assert isinstance(error, HSMError)
    assert str(error) == "executor failed"


# -----------------------------------------------------------------------------
# STATE ERROR TESTS
# -----------------------------------------------------------------------------


def test_state_error():
    """Test StateError functionality."""
    state_info = {"state_id": "state1", "data": {"key": "value"}}
    error = StateError("state error", state_info)
    assert error.state_id == "state1"
    assert error.data == {"key": "value"}


def test_invalid_transition_error():
    """Test InvalidTransitionError functionality."""
    error = InvalidTransitionError(
        "invalid transition", "source_state", "target_state", "event_data", {"detail": "value"}
    )
    assert error.source_state == "source_state"
    assert error.target_state == "target_state"
    assert error.event == "event_data"
    assert error.details == {"detail": "value"}


def test_invalid_state_error():
    """Test InvalidStateError functionality."""
    error = InvalidStateError("invalid state", "state1", "operation1", {"detail": "value"})
    assert error.state_id == "state1"
    assert error.operation == "operation1"
    assert error.details == {"detail": "value"}


# -----------------------------------------------------------------------------
# VALIDATION ERROR TESTS
# -----------------------------------------------------------------------------


def test_validation_error():
    """Test ValidationError functionality."""
    validation_results = ["error1", "error2"]
    error = ValidationError("validation failed", "test_component", validation_results, {"detail": "value"})
    assert error.component == "test_component"
    assert error.validation_results == validation_results
    assert error.details == {"detail": "value"}


def test_configuration_error():
    """Test ConfigurationError functionality."""
    validation_errors = {"field1": "error1", "field2": "error2"}
    error = ConfigurationError("config error", "test_component", validation_errors, {"detail": "value"})
    assert error.component == "test_component"
    assert error.validation_errors == validation_errors
    assert error.details == {"detail": "value"}


# -----------------------------------------------------------------------------
# EXECUTION ERROR TESTS
# -----------------------------------------------------------------------------


def test_execution_error():
    """Test ExecutionError functionality."""
    execution_info = {"operation": "test_op", "status": "failed"}
    error = ExecutionError("execution error", execution_info, {"detail": "value"})
    assert error.operation == "test_op"
    assert error.status == "failed"
    assert error.details == {"detail": "value"}


def test_guard_evaluation_error(sample_event, sample_state_data):
    """Test GuardEvaluationError functionality."""
    error = GuardEvaluationError("guard failed", "TestGuard", sample_state_data, sample_event, {"detail": "value"})
    assert error.guard_name == "TestGuard"
    assert error.state_data == sample_state_data
    assert error.event == sample_event
    assert error.details == {"detail": "value"}


def test_action_execution_error(sample_event, sample_state_data):
    """Test ActionExecutionError functionality."""
    error = ActionExecutionError("action failed", "TestAction", sample_state_data, sample_event, {"detail": "value"})
    assert error.action_name == "TestAction"
    assert error.state_data == sample_state_data
    assert error.event == sample_event
    assert error.details == {"detail": "value"}


# -----------------------------------------------------------------------------
# CONCURRENCY ERROR TESTS
# -----------------------------------------------------------------------------


def test_concurrency_error():
    """Test ConcurrencyError functionality."""
    error = ConcurrencyError("concurrency error", "test_operation", "test_resource", {"detail": "value"})
    assert error.operation == "test_operation"
    assert error.resource == "test_resource"
    assert error.details == {"detail": "value"}


# -----------------------------------------------------------------------------
# QUEUE ERROR TESTS
# -----------------------------------------------------------------------------


def test_event_queue_full_error():
    """Test EventQueueFullError functionality."""
    error = EventQueueFullError("queue full", 10, 20, "dropped_event", {"detail": "value"})
    assert error.queue_size == 10
    assert error.max_size == 20
    assert error.dropped_event == "dropped_event"
    assert error.details == {"detail": "value"}


# -----------------------------------------------------------------------------
# ERROR CONTEXT TESTS
# -----------------------------------------------------------------------------


def test_error_context():
    """Test ErrorContext functionality."""
    error = HSMError("test error", {"detail": "value"})
    context = create_error_context(error, "test traceback")

    assert context.error_type == HSMError
    assert isinstance(context.timestamp, float)
    assert context.traceback == "test traceback"
    assert context.details == {"detail": "value"}


def test_error_context_immutability():
    """Test that ErrorContext is immutable."""
    error = HSMError("test error")
    context = create_error_context(error, "test traceback")

    with pytest.raises(Exception):  # dataclass is frozen
        context.traceback = "new traceback"


# -----------------------------------------------------------------------------
# ERROR CHAINING TESTS
# -----------------------------------------------------------------------------


def test_error_chaining():
    """Test error chaining functionality."""
    try:
        try:
            raise ValueError("inner error")
        except ValueError as ve:
            raise HSMError("outer error") from ve
    except HSMError as he:
        assert isinstance(he.__cause__, ValueError)
        assert str(he.__cause__) == "inner error"


def test_error_with_cause():
    """Test error creation with explicit cause."""
    cause = ValueError("cause error")
    error = GuardEvaluationError("guard failed", "TestGuard", {}, None, {"cause": str(cause)})
    assert error.details["cause"] == str(cause)


# -----------------------------------------------------------------------------
# EDGE CASES AND SPECIAL VALUES
# -----------------------------------------------------------------------------


def test_errors_with_none_values():
    """Test error creation with None values."""
    # HSMError with None details
    error = HSMError("test message", None)
    assert error.details == {}

    # ValidationError with None values
    error = ValidationError("test message", None, None, None)
    assert error.component is None
    assert error.validation_results == []
    assert error.details == {}

    # ConfigurationError with None values
    error = ConfigurationError("test message", "component", None, None)
    assert error.validation_errors == {}
    assert error.details == {}


def test_errors_with_empty_values():
    """Test error creation with empty values."""
    # Empty string message
    error = HSMError("")
    assert error.message == ""

    # Empty details
    error = HSMError("message", {})
    assert error.details == {}

    # Empty validation results
    error = ValidationError("message", "component", [], {})
    assert error.validation_results == []


def test_large_error_data():
    """Test errors with large data structures."""
    # Create a large nested dictionary
    large_data = {"key" + str(i): "x" * 1000 for i in range(1000)}
    nested_data = {"level1": {"level2": {"level3": large_data}}}

    error = HSMError("test message", nested_data)
    assert error.details == nested_data

    # Test with large validation results
    large_results = ["error" + str(i) for i in range(1000)]
    error = ValidationError("test message", "component", large_results)
    assert len(error.validation_results) == 1000
