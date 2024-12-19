# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import Any, Dict, List, Optional

import pytest

from hsm.core.errors import ValidationError
from hsm.core.validation import ValidationContext, ValidationRule, ValidationSeverity, Validator
from hsm.interfaces.abc import AbstractAction, AbstractGuard, AbstractState, AbstractTransition
from hsm.interfaces.types import StateID, ValidationResult

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class MockState(AbstractState):
    def __init__(self, state_id: str, data: Optional[Dict[str, Any]] = None):
        self._id = state_id
        self._data = data if data is not None else {}

    def get_id(self) -> StateID:
        return self._id

    @property
    def data(self) -> Dict[str, Any]:
        return self._data  # Return the actual dictionary reference


class MockTransition(AbstractTransition):
    def __init__(
        self,
        source_id: str,
        target_id: str,
        guard: Optional[AbstractGuard] = None,
        actions: Optional[List[AbstractAction]] = None,
        priority: int = 0,
    ):
        self._source_id = source_id
        self._target_id = target_id
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state_id(self) -> StateID:
        return self._source_id

    def get_target_state_id(self) -> StateID:
        return self._target_id

    def get_guard(self) -> Optional[AbstractGuard]:
        return self._guard

    def get_actions(self) -> List[AbstractAction]:
        return self._actions

    def get_priority(self) -> int:
        return self._priority


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def basic_states() -> List[AbstractState]:
    return [
        MockState("state1"),
        MockState("state2"),
        MockState("state3"),
    ]


@pytest.fixture
def basic_transitions(basic_states: List[AbstractState]) -> List[AbstractTransition]:
    return [
        MockTransition("state1", "state2"),
        MockTransition("state2", "state3"),
        MockTransition("state3", "state1"),
    ]


@pytest.fixture
def basic_validator(basic_states: List[AbstractState], basic_transitions: List[AbstractTransition]) -> Validator:
    return Validator(basic_states, basic_transitions, basic_states[0])


@pytest.fixture
def validation_context(
    basic_states: List[AbstractState], basic_transitions: List[AbstractTransition]
) -> ValidationContext:
    return ValidationContext(basic_states, basic_transitions, basic_states[0])


# -----------------------------------------------------------------------------
# VALIDATION RULE TESTS
# -----------------------------------------------------------------------------


def test_validation_rule_immutability() -> None:
    """Test that ValidationRule instances are immutable."""
    rule = ValidationRule("test", lambda x: True, ValidationSeverity.ERROR, "Test rule")

    with pytest.raises(Exception):  # dataclass is frozen
        rule.name = "new_name"  # type: ignore

    with pytest.raises(Exception):
        rule.severity = ValidationSeverity.WARNING  # type: ignore


def test_validation_context_result_addition(validation_context: ValidationContext) -> None:
    """Test adding validation results."""
    validation_context.add_result("ERROR", "Test message", {"key": "value"})

    assert len(validation_context.current_results) == 1
    result = validation_context.current_results[0]
    assert result.severity == "ERROR"
    assert result.message == "Test message"
    assert result.context == {"key": "value"}


# -----------------------------------------------------------------------------
# VALIDATOR INITIALIZATION TESTS
# -----------------------------------------------------------------------------


def test_validator_initialization_empty_states() -> None:
    """Test validator initialization with empty states list."""
    with pytest.raises(ValidationError, match="State machine must have at least one state"):
        Validator([], [], MockState("initial"))


def test_validator_initialization_invalid_initial_state(basic_states: List[AbstractState]) -> None:
    """Test validator initialization with invalid initial state."""
    invalid_initial = MockState("invalid")
    with pytest.raises(ValidationError, match="Initial state must be in states list"):
        Validator(basic_states, [], invalid_initial)


def test_validation_empty_transitions(basic_states: List[AbstractState]) -> None:
    """Test validation with no transitions."""
    validator = Validator(basic_states, [], basic_states[0])

    results = validator.validate_structure()
    assert any(r.severity == "ERROR" and "reachable" in r.message for r in results)


def test_validation_duplicate_state_ids() -> None:
    """Test validation of duplicate state IDs."""
    states = [
        MockState("state1"),
        MockState("state1"),  # Duplicate ID
    ]
    validator = Validator(states, [], states[0])

    results = validator.validate_structure()
    assert any(r.severity == "ERROR" and "unique" in r.message.lower() for r in results)
