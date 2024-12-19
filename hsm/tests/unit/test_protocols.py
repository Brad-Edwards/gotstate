# hsm/tests/test_protocols.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
"""Test suite for protocol conformance and type checking.

This module contains unit tests for protocol conformance and type checking
of the core protocols in the HSM implementation.
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
from hsm.tests.utils import (
    EVENT_ID_TEST_CASES,
    PAYLOAD_TEST_CASES,
    PRIORITY_TEST_CASES,
    ValidEvent,
    ValidTransition,
    create_test_event,
)


# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------
class MockGuard(AbstractGuard):
    def check(self, event: AbstractEvent, state_data: Any) -> bool:
        return True


class MockAction(AbstractAction):
    def execute(self, event: AbstractEvent, state_data: Any) -> None:
        pass


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
    """Event with wrong return type for get_id."""

    def get_id(self) -> int:
        return 123  # Incorrect type, expecting a str alias

    def get_payload(self) -> Any:
        return "payload"

    def get_priority(self) -> int:
        return 1


class InvalidGuard(AbstractGuard):
    """Guard that returns a string instead of bool."""

    def check(self, event: AbstractEvent, state_data: Any) -> bool:  # NOSONAR
        return "not a bool"  # type: ignore


class InvalidValidator(AbstractValidator):
    """Validator that returns the wrong type."""

    def validate_structure(self) -> List[ValidationResult]:
        return ["invalid"]  # type: ignore

    def validate_behavior(self) -> List[ValidationResult]:
        return []

    def validate_data(self) -> List[ValidationResult]:
        return []


# -----------------------------------------------------------------------------
# PROTOCOL CONFORMANCE TESTS
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


def test_event_protocol_type_safety() -> None:
    """Test Event protocol type safety."""
    for invalid_id in EVENT_ID_TEST_CASES["invalid"]:
        with pytest.raises(TypeError):
            ValidEvent(invalid_id)  # type: ignore

    # Test invalid priority types
    invalid_priorities = [None, "1", 1.0, True]
    for invalid_priority in invalid_priorities:
        with pytest.raises(TypeError):
            ValidEvent("test", priority=invalid_priority)  # type: ignore


def test_event_protocol_payload_handling() -> None:
    """Test Event protocol with various payload types."""
    for case_name, payloads in PAYLOAD_TEST_CASES.items():
        for payload in payloads:
            event = create_test_event(payload=payload)
            assert event.get_payload() == payload


def test_event_protocol_priority_values() -> None:
    """Test Event protocol with various priority values."""
    for case_name, priorities in PRIORITY_TEST_CASES.items():
        for priority in priorities:
            event = create_test_event(priority=priority)
            assert event.get_priority() == priority
            assert isinstance(event.get_priority(), int)


def test_event_protocol_special_ids() -> None:
    """Test Event protocol with special event IDs."""
    for special_id in EVENT_ID_TEST_CASES["special"]:
        event = create_test_event(event_id=special_id)
        assert event.get_id() == special_id


def test_validator_protocol_conformance() -> None:
    """Test validator protocol conformance."""
    validator = InvalidValidator()
    # Test that it implements the protocol methods
    assert hasattr(validator, "validate_structure")
    assert hasattr(validator, "validate_behavior")
    assert hasattr(validator, "validate_data")
    # Test method return types
    assert isinstance(validator.validate_structure(), list)
    assert isinstance(validator.validate_behavior(), list)
    assert isinstance(validator.validate_data(), list)


def test_guard_protocol_conformance() -> None:
    """Test guard protocol conformance."""
    guard = MockGuard()
    # Test that it implements the protocol methods
    assert hasattr(guard, "check")
    # Test method return type
    assert isinstance(guard.check(create_test_event(), {}), bool)
