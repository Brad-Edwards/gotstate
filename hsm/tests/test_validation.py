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

    def on_enter(self) -> None:
        pass

    def on_exit(self) -> None:
        pass


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


def test_validation_rule_equality() -> None:
    """Test that identical ValidationRules are equal."""
    check_func = lambda x: True
    rule1 = ValidationRule("test", check_func, ValidationSeverity.ERROR, "Test rule")
    rule2 = ValidationRule("test", check_func, ValidationSeverity.ERROR, "Test rule")

    assert rule1 == rule2


# -----------------------------------------------------------------------------
# VALIDATION CONTEXT TESTS
# -----------------------------------------------------------------------------


def test_validation_context_initialization(validation_context: ValidationContext) -> None:
    """Test ValidationContext initialization and basic properties."""
    assert len(validation_context.states) == 3
    assert len(validation_context.transitions) == 3
    assert validation_context.initial_state is not None
    assert len(validation_context.current_results) == 0


def test_validation_context_state_lookup(validation_context: ValidationContext) -> None:
    """Test state lookup functionality."""
    state = validation_context.get_state_by_id("state1")
    assert state is not None
    assert state.get_id() == "state1"

    assert validation_context.get_state_by_id("nonexistent") is None


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


def test_validator_initialization_success(basic_validator: Validator) -> None:
    """Test successful validator initialization."""
    assert basic_validator is not None
    assert isinstance(basic_validator, Validator)


# -----------------------------------------------------------------------------
# RULE REGISTRATION TESTS
# -----------------------------------------------------------------------------


def test_add_rule_invalid_type(basic_validator: Validator) -> None:
    """Test adding a rule with invalid rule type."""
    with pytest.raises(ValidationError, match="Invalid rule type"):
        basic_validator.add_rule("test", lambda ctx: True, ValidationSeverity.ERROR, "Test rule", "invalid_type")


def test_add_rule_duplicate_name(basic_validator: Validator) -> None:
    """Test adding a rule with duplicate name."""
    basic_validator.add_rule("test", lambda ctx: True, ValidationSeverity.ERROR, "Test rule")

    with pytest.raises(ValidationError, match="Duplicate rule name"):
        basic_validator.add_rule("test", lambda ctx: True, ValidationSeverity.ERROR, "Test rule")


def test_add_rule_success(basic_validator: Validator) -> None:
    """Test successful rule addition."""
    basic_validator.add_rule("test", lambda ctx: True, ValidationSeverity.ERROR, "Test rule")

    # Validate by running structural validation
    results = basic_validator.validate_structure()
    assert len(results) == 0  # Rule passed


# -----------------------------------------------------------------------------
# VALIDATION EXECUTION TESTS
# -----------------------------------------------------------------------------


def test_validate_structure_orphan_states(basic_states: List[AbstractState]) -> None:
    """Test validation of orphaned states."""
    # Create transitions that leave one state unreachable
    transitions = [MockTransition("state1", "state2")]
    validator = Validator(basic_states, transitions, basic_states[0])

    results = validator.validate_structure()
    assert any(r.severity == "ERROR" and "reachable" in r.message for r in results)


def test_validate_structure_invalid_transitions(basic_states: List[AbstractState]) -> None:
    """Test validation of invalid transitions."""
    transitions = [MockTransition("state1", "nonexistent")]
    validator = Validator(basic_states, transitions, basic_states[0])

    results = validator.validate_structure()
    assert any(r.severity == "ERROR" and "valid states" in r.message for r in results)


def test_validate_behavior_guard_safety(basic_states: List[AbstractState]) -> None:
    """Test validation of guard safety."""

    class InvalidGuard:  # Not implementing AbstractGuard
        def invalid_method(self, event: Any, state_data: Any) -> bool:
            return True

    transitions = [MockTransition("state1", "state2", guard=InvalidGuard())]  # type: ignore
    validator = Validator(basic_states, transitions, basic_states[0])

    results = validator.validate_behavior()
    assert any(r.severity == "WARNING" and "guard" in r.message.lower() for r in results)


def test_validate_data_isolation(basic_validator: Validator) -> None:
    """Test validation of state data isolation."""
    shared_data = {}
    states = [
        MockState("state1", shared_data),
        MockState("state2", shared_data),  # Using same data dict
    ]
    validator = Validator(states, [], states[0])

    # Add a specific data isolation rule with rule_type="data"
    validator.add_rule(
        "test_data_isolation",
        validator._check_data_isolation,
        ValidationSeverity.ERROR,
        "State data must be isolated",
        rule_type="data",  # Specify rule type as "data"
    )

    results = validator.validate_data()

    # Debug output
    print(f"Number of results: {len(results)}")
    for r in results:
        print(f"Severity: {r.severity}, Message: {r.message}")

    # Check for data isolation violation
    assert any(
        r.severity == "ERROR" and "shares data dictionary" in r.message for r in results
    ), "Data isolation violation not detected"


# -----------------------------------------------------------------------------
# EDGE CASES AND ERROR HANDLING
# -----------------------------------------------------------------------------


def test_validation_rule_exception_handling(basic_validator: Validator) -> None:
    """Test handling of exceptions in validation rules."""

    def failing_rule(context: ValidationContext) -> bool:
        raise RuntimeError("Rule failed")

    basic_validator.add_rule("failing_rule", failing_rule, ValidationSeverity.ERROR, "This rule always fails")

    results = basic_validator.validate_structure()
    assert any(
        r.severity == "ERROR" and "failed with exception" in r.message and r.context.get("error") == "Rule failed"
        for r in results
    )


def test_validation_empty_transitions(basic_states: List[AbstractState]) -> None:
    """Test validation with no transitions."""
    validator = Validator(basic_states, [], basic_states[0])

    results = validator.validate_structure()
    assert any(r.severity == "ERROR" and "reachable" in r.message for r in results)


def test_validation_cyclic_transitions(basic_validator: Validator) -> None:
    """Test validation of cyclic transitions."""
    results = basic_validator.validate_structure()
    assert not any(r.severity == "ERROR" for r in results)  # Cycles are allowed


def test_validation_duplicate_state_ids() -> None:
    """Test validation of duplicate state IDs."""
    states = [
        MockState("state1"),
        MockState("state1"),  # Duplicate ID
    ]
    validator = Validator(states, [], states[0])

    results = validator.validate_structure()
    assert any(r.severity == "ERROR" and "unique" in r.message.lower() for r in results)


# -----------------------------------------------------------------------------
# INTEGRATION TESTS
# -----------------------------------------------------------------------------


def test_full_validation_workflow(basic_validator: Validator) -> None:
    """Test complete validation workflow."""
    # Run all validation types
    structure_results = basic_validator.validate_structure()
    behavior_results = basic_validator.validate_behavior()
    data_results = basic_validator.validate_data()

    # Verify results
    assert isinstance(structure_results, list)
    assert isinstance(behavior_results, list)
    assert isinstance(data_results, list)

    # Verify no structural errors in basic valid configuration
    assert not any(r.severity == "ERROR" for r in structure_results)


def test_validation_result_aggregation(basic_validator: Validator) -> None:
    """Test aggregation of validation results across multiple rules."""
    # Add multiple rules that will fail
    basic_validator.add_rule("always_fail_1", lambda ctx: False, ValidationSeverity.ERROR, "First failing rule")
    basic_validator.add_rule("always_fail_2", lambda ctx: False, ValidationSeverity.WARNING, "Second failing rule")

    results = basic_validator.validate_structure()

    # Verify both failures are reported
    assert len(results) == 2
    assert any(r.severity == "ERROR" for r in results)
    assert any(r.severity == "WARNING" for r in results)


def test_validation_context_isolation(basic_validator: Validator) -> None:
    """Test that validation contexts are properly isolated."""
    # Run structure validation
    structure_results = basic_validator.validate_structure()

    # Run behavior validation
    behavior_results = basic_validator.validate_behavior()

    # Verify results don't interfere
    assert len(basic_validator._context.current_results) == len(behavior_results)
    assert structure_results is not behavior_results
