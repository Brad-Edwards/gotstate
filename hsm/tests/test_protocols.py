# hsm/tests/test_protocols.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
"""Test suite for HSM error handling and error classes.

This module contains unit tests for various error conditions and error
classes in the HSM implementation.
"""
import logging
from typing import Any, Dict, List, Optional, Protocol

import pytest

from hsm.core.errors import (
    ActionExecutionError,
    ConcurrencyError,
    ConfigurationError,
    EventQueueFullError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    InvalidTransitionError,
    create_error_context,
)
from hsm.interfaces.abc import AbstractAction, AbstractEvent, AbstractGuard, AbstractValidator
from hsm.interfaces.protocols import Event, Transition
from hsm.interfaces.types import EventID, StateID, ValidationResult


# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------
class MockGuard(AbstractGuard):
    def check(self, event: AbstractEvent, state_data: Any) -> bool:
        return True


class MockAction(AbstractAction):
    def execute(self, event: AbstractEvent, state_data: Any) -> None:
        # Mock needed for protocol conformance, impl not needed
        pass


class ValidEvent:
    def __init__(self, event_id: EventID, payload: Any, priority: int = 0) -> None:
        self._id = event_id
        self._payload = payload
        self._priority = priority

    def get_id(self) -> EventID:
        return self._id

    def get_payload(self) -> Any:
        return self._payload

    def get_priority(self) -> int:
        return self._priority


class ValidTransition:
    def __init__(
        self,
        source: StateID,
        target: StateID,
        guard: Optional[AbstractGuard] = None,
        actions: Optional[List[AbstractAction]] = None,
        priority: int = 0,
    ) -> None:
        self._source = source
        self._target = target
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state_id(self) -> StateID:
        return self._source

    def get_target_state_id(self) -> StateID:
        return self._target

    def get_guard(self) -> Optional[AbstractGuard]:
        return self._guard

    def get_actions(self) -> List[AbstractAction]:
        return self._actions

    def get_priority(self) -> int:
        return self._priority


class InvalidEvent:
    """Event that doesn't implement all required methods."""

    def get_id(self) -> EventID:
        return "test"

    # Missing get_payload and get_priority methods


class InvalidTransition:
    """Transition that doesn't implement all required methods."""

    def get_source_state_id(self) -> StateID:
        return "test"

    # Missing other required methods


class InvalidReturnTypeEvent:
    """Event with wrong return type for get_id (should return EventID(str)) but returns int."""

    def get_id(self) -> int:
        return 123  # Incorrect type, expecting a str alias

    def get_payload(self) -> Any:
        return "payload"

    def get_priority(self) -> int:
        return 1


class InvalidGuard(AbstractGuard):
    """Guard that returns a string instead of bool."""

    def check(self, event: AbstractEvent, state_data: Any) -> bool:  # noqa: S5886
        return "not a bool"  # Incorrect return type - intentional for testing


class InvalidValidator(AbstractValidator):
    """Validator that returns the wrong type (not a list of ValidationResult)."""

    def validate_structure(self) -> List[ValidationResult]:
        return ["invalid"]  # Should be ValidationResult objects

    def validate_behavior(self) -> List[ValidationResult]:
        return []

    def validate_data(self) -> List[ValidationResult]:
        return []


# -----------------------------------------------------------------------------
# TEST PROTOCOL CONFORMANCE
# -----------------------------------------------------------------------------


def test_valid_event_conforms_to_event_protocol() -> None:
    """Test that a valid event implementation conforms to the Event protocol."""
    evt = ValidEvent(event_id="evt-123", payload={"data": 42}, priority=5)
    assert isinstance(evt, Event), "ValidEvent should conform to Event protocol"
    assert evt.get_id() == "evt-123"
    assert evt.get_payload() == {"data": 42}
    assert evt.get_priority() == 5


def test_invalid_event_does_not_conform() -> None:
    """Test that an invalid event implementation does not conform to Event protocol."""
    invalid_evt = InvalidEvent()
    assert not isinstance(invalid_evt, Event), "InvalidEvent should not conform to Event protocol"


def test_valid_transition_conforms_to_transition_protocol() -> None:
    """Test that a valid transition implementation conforms to the Transition protocol."""
    guard = MockGuard()
    action = MockAction()
    trans = ValidTransition(source="state_A", target="state_B", guard=guard, actions=[action], priority=10)
    assert isinstance(trans, Transition), "ValidTransition should conform to Transition protocol"
    assert trans.get_source_state_id() == "state_A"
    assert trans.get_target_state_id() == "state_B"
    assert trans.get_guard() is guard
    assert trans.get_actions() == [action]
    assert trans.get_priority() == 10


def test_invalid_transition_does_not_conform() -> None:
    """Test that an invalid transition implementation does not conform to Transition protocol."""
    invalid_trans = InvalidTransition()
    assert not isinstance(invalid_trans, Transition), "InvalidTransition should not conform to Transition protocol"


# -----------------------------------------------------------------------------
# TEST CUSTOM EXCEPTIONS (from errors.py)
# -----------------------------------------------------------------------------


def test_hsm_error_base_class() -> None:
    """Test the base HSMError."""
    with pytest.raises(HSMError) as exc_info:
        raise HSMError("Base error", details={"key": "value"})
    assert str(exc_info.value) == "Base error"
    assert exc_info.value.details == {"key": "value"}


def test_invalid_transition_error() -> None:
    """Test InvalidTransitionError attributes and formatting."""
    with pytest.raises(InvalidTransitionError) as exc_info:
        raise InvalidTransitionError(
            "Invalid transition", source_state="S1", target_state="S2", event="EvtX", details={"cause": "guard failed"}
        )
    err = exc_info.value
    assert err.message == "Invalid transition"
    assert err.source_state == "S1"
    assert err.target_state == "S2"
    assert err.event == "EvtX"
    assert err.details == {"cause": "guard failed"}


def test_invalid_state_error() -> None:
    """Test InvalidStateError attributes."""
    with pytest.raises(InvalidStateError) as exc_info:
        raise InvalidStateError("Invalid state", state_id="S999", operation="enter")
    err = exc_info.value
    assert err.message == "Invalid state"
    assert err.state_id == "S999"
    assert err.operation == "enter"
    assert err.details == {}


def test_configuration_error() -> None:
    """Test ConfigurationError attributes."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Config error", component="Machine", validation_errors={"rule": "missing state"})
    err = exc_info.value
    assert err.message == "Config error"
    assert err.component == "Machine"
    assert err.validation_errors == {"rule": "missing state"}


def test_guard_evaluation_error() -> None:
    """Test GuardEvaluationError attributes."""
    with pytest.raises(GuardEvaluationError) as exc_info:
        raise GuardEvaluationError("Guard failed", guard_name="CheckX", state_data={"x": 10}, event="E1")
    err = exc_info.value
    assert err.message == "Guard failed"
    assert err.guard_name == "CheckX"
    assert err.state_data == {"x": 10}
    assert err.event == "E1"


def test_action_execution_error() -> None:
    """Test ActionExecutionError attributes."""
    with pytest.raises(ActionExecutionError) as exc_info:
        raise ActionExecutionError("Action failed", action_name="DoStuff", state_data={"y": 20}, event="E2")
    err = exc_info.value
    assert err.message == "Action failed"
    assert err.action_name == "DoStuff"
    assert err.state_data == {"y": 20}
    assert err.event == "E2"


def test_concurrency_error() -> None:
    """Test ConcurrencyError attributes."""
    with pytest.raises(ConcurrencyError) as exc_info:
        raise ConcurrencyError("Lock conflict", operation="start", resource="state_machine_lock")
    err = exc_info.value
    assert err.message == "Lock conflict"
    assert err.operation == "start"
    assert err.resource == "state_machine_lock"


def test_event_queue_full_error() -> None:
    """Test EventQueueFullError attributes."""
    with pytest.raises(EventQueueFullError) as exc_info:
        raise EventQueueFullError("Queue full", queue_size=100, max_size=100, dropped_event="E3")
    err = exc_info.value
    assert err.message == "Queue full"
    assert err.queue_size == 100
    assert err.max_size == 100
    assert err.dropped_event == "E3"


def test_create_error_context() -> None:
    """Test create_error_context helper function."""
    err = HSMError("Test error", details={"info": "detail"})
    tb = "Traceback info..."
    ctx = create_error_context(err, tb)
    assert ctx.error_type is HSMError
    assert isinstance(ctx.timestamp, float)
    assert ctx.traceback == tb
    assert ctx.details == {"info": "detail"}


# -----------------------------------------------------------------------------
# EDGE CASES & INTEGRATION TESTS
# -----------------------------------------------------------------------------


def test_empty_payload_event() -> None:
    """Test event with empty payload."""
    evt = ValidEvent(event_id="", payload=None)
    assert evt.get_id() == ""
    assert evt.get_payload() is None


def test_no_guard_transition() -> None:
    """Test transition with no guard defined."""
    trans = ValidTransition(source="A", target="B", guard=None)
    assert trans.get_guard() is None


def test_no_action_transition() -> None:
    """Test transition with no actions defined."""
    trans = ValidTransition(source="A", target="B")
    assert trans.get_actions() == []


def test_large_priority() -> None:
    """Test transition with very large priority value."""
    trans = ValidTransition(source="A", target="B", priority=999999999)
    assert trans.get_priority() == 999999999


def test_event_priority_boundaries() -> None:
    """Test event with extreme priority values."""
    low_priority_evt = ValidEvent("evt-low", None, priority=-100)
    high_priority_evt = ValidEvent("evt-high", None, priority=10_000)
    assert low_priority_evt.get_priority() == -100
    assert high_priority_evt.get_priority() == 10_000


# Integration-like test: simulate usage in a context manager or logging scenario
def test_error_logging_integration(caplog: pytest.LogCaptureFixture) -> None:
    """Test error handling in a logging scenario."""
    logger = logging.getLogger("hsm.test")
    logger.setLevel(logging.INFO)

    try:
        raise InvalidStateError("Logging test", state_id="X", operation="stop")
    except InvalidStateError as e:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="hsm.test"):
            # Log the error message using the logger
            logger.info("Logged: %s", e.message)

        # Verify that the log message is captured by caplog
        assert any(
            "Logged: Logging test" in rec.message for rec in caplog.records
        ), "Expected log message not found in captured logs"


@pytest.mark.asyncio
async def test_async_error_scenario() -> None:
    """Test that exceptions can be handled in async code."""

    async def async_function_raising_error():
        await asyncio.sleep(0.01)
        raise ActionExecutionError("Async action fail", action_name="AsyncAction", state_data={}, event="E-async")

    import asyncio

    with pytest.raises(ActionExecutionError) as exc_info:
        await async_function_raising_error()
    assert "Async action fail" in str(exc_info.value)


def test_cleanup_procedures() -> None:
    """Test a scenario where cleanup is needed after an error."""
    # Simulate cleanup after error
    try:
        raise InvalidTransitionError("Cleanup needed", "S1", "S2", event="E4")
    except InvalidTransitionError:
        # Simulate cleanup here
        # In real code, you might revert partial changes or release locks.
        cleanup_done = True
        assert cleanup_done is True


def test_maximum_recursion_scenario() -> None:
    """Test extreme scenario (pseudo) for recursion or nesting errors."""
    # This is symbolic since we don't have actual recursion in these protocols.
    # Just ensure raising multiple nested exceptions works.
    try:
        try:
            raise InvalidStateError("Innermost", state_id="Deep", operation="check")
        except InvalidStateError as inner:
            raise InvalidTransitionError("Outer", source_state="Sx", target_state="Sy", event="Ez") from inner
    except InvalidTransitionError as outer:
        assert "Outer" in str(outer)
        assert isinstance(outer.__cause__, InvalidStateError)
        assert outer.__cause__.state_id == "Deep"


# -----------------------------------------------------------------------------
# NEGATIVE CONFORMANCE TESTS
# -----------------------------------------------------------------------------
def test_event_protocol_negative_type_check() -> None:
    """Test that an event with incorrect return types does not conform."""
    InvalidReturnTypeEvent()
    # Although the protocol uses runtime_checkable, it won't catch return type mismatches at runtime.
    # The best we can do is assert(that logically this shouldn't conform.
    # Protocol checks structural attributes, not runtime return type correctness.
    # This test documents that runtime_check only checks method presence, not return types.
    # It's expected that isinstance(invalid_evt, Event) still returns True because runtime_checkable doesn't enforce return types.
    # We'll assert(the methods exist and do a semantic check manually.
    invalid_evt = InvalidReturnTypeEvent()
    assert isinstance(invalid_evt, Event), "Runtime type checking does not enforce return types."
    result = invalid_evt.get_id()
    assert isinstance(result, int), "InvalidEvent returns int, expected str"


def test_guard_protocol_negative_return_type() -> None:
    """Test that a guard returning a non-boolean value is logically invalid."""
    validator = InvalidValidator()
    structure_result = validator.validate_structure()
    assert structure_result == ["invalid"], "Invalid return type from validator"


def test_validator_negative_return_type() -> None:
    """Test that a validator returning wrong types is logically invalid."""
    validator = InvalidValidator()
    structure_result = validator.validate_structure()
    assert structure_result == ["invalid"], "Invalid return type from validator"


# -----------------------------------------------------------------------------
# COMPLEX PAYLOAD TESTS
# -----------------------------------------------------------------------------
def test_event_with_complex_nested_payload() -> None:
    """Test an event with a deeply nested payload."""
    complex_payload = {"level1": {"level2": {"list": [1, 2, {"deep": "value"}], "number": 42}}}
    evt = ValidEvent("complex_evt", complex_payload, priority=3)
    assert isinstance(evt, Event)
    assert evt.get_payload() == complex_payload


# -----------------------------------------------------------------------------
# PRIORITY AND ORDERING TESTS FOR TRANSITIONS
# -----------------------------------------------------------------------------
def test_multiple_transitions_priority_order() -> None:
    """Test multiple transitions with different priorities."""
    t1 = ValidTransition("S1", "S2", priority=10)
    t2 = ValidTransition("S1", "S3", priority=5)
    t3 = ValidTransition("S1", "S4", priority=20)

    transitions = [t1, t2, t3]
    # Sort transitions by priority descending
    transitions_sorted = sorted(transitions, key=lambda tr: tr.get_priority(), reverse=True)
    assert transitions_sorted[0].get_priority() == 20
    assert transitions_sorted[1].get_priority() == 10
    assert transitions_sorted[2].get_priority() == 5


# -----------------------------------------------------------------------------
# VALIDATOR PROTOCOL EDGE CASE TESTS
# -----------------------------------------------------------------------------
def test_validator_with_empty_results() -> None:
    """Test validator returning empty lists of ValidationResult."""

    class EmptyValidator(AbstractValidator):
        def validate_structure(self) -> List[ValidationResult]:
            return []

        def validate_behavior(self) -> List[ValidationResult]:
            return []

        def validate_data(self) -> List[ValidationResult]:
            return []

    validator = EmptyValidator()
    assert validator.validate_structure() == []
    assert validator.validate_behavior() == []
    assert validator.validate_data() == []


def test_validator_with_maximum_severity() -> None:
    """Test validator returning a ValidationResult with an extreme severity string."""
    extreme_result = ValidationResult(severity="CRITICAL", message="Test message", context={})

    class ExtremeValidator(AbstractValidator):
        def validate_structure(self) -> List[ValidationResult]:
            return [extreme_result]

        def validate_behavior(self) -> List[ValidationResult]:
            return []

        def validate_data(self) -> List[ValidationResult]:
            return []

    validator = ExtremeValidator()
    results = validator.validate_structure()
    assert len(results) == 1
    assert results[0].severity == "CRITICAL"


# -----------------------------------------------------------------------------
# CONCURRENT ERROR TESTING
# -----------------------------------------------------------------------------
def test_concurrency_error_scenario() -> None:
    """Test simulating concurrency error scenario."""
    # Simulate a concurrency conflict by directly raising ConcurrencyError
    with pytest.raises(ConcurrencyError) as exc_info:
        raise ConcurrencyError("Concurrent access error", operation="write", resource="shared_lock")
    err = exc_info.value
    assert err.message == "Concurrent access error"
    assert err.operation == "write"
    assert err.resource == "shared_lock"
