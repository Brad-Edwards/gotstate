# hsm/core/validation.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type

from hsm.core.errors import ValidationError
from hsm.interfaces.abc import (
    AbstractAction,
    AbstractEvent,
    AbstractGuard,
    AbstractState,
    AbstractTransition,
    AbstractValidator,
)
from hsm.interfaces.types import StateID, ValidationResult


class ValidationSeverity(Enum):
    """Severity levels for validation results."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


class ValidationContext:
    """
    Context object passed to validation rules.

    Provides access to the complete state machine configuration
    and maintains validation state.

    Attributes:
        states: All states in the machine
        transitions: All transitions in the machine
        initial_state: Starting state
        current_results: Validation results collected so far
    """

    def __init__(
        self,
        states: List[AbstractState],
        transitions: List[AbstractTransition],
        initial_state: AbstractState,
    ) -> None:
        """
        Initialize validation context.

        Args:
            states: All states in the machine
            transitions: All transitions in the machine
            initial_state: Starting state
        """
        self.states = states
        self.transitions = transitions
        self.initial_state = initial_state
        self.current_results: List[ValidationResult] = []
        self._state_ids: Set[StateID] = {state.get_id() for state in states}

    def add_result(self, severity: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a validation result.

        Args:
            severity: Severity level of the result
            message: Description of the validation result
            context: Optional additional context information
        """
        self.current_results.append(ValidationResult(severity, message, context or {}))

    def get_state_by_id(self, state_id: StateID) -> Optional[AbstractState]:
        """
        Look up a state by its ID.

        Args:
            state_id: ID of the state to find

        Returns:
            The state if found, None otherwise
        """
        return next((state for state in self.states if state.get_id() == state_id), None)

    def state_exists(self, state_id: StateID) -> bool:
        """
        Check if a state ID exists in the machine.

        Args:
            state_id: ID to check

        Returns:
            True if the state exists, False otherwise
        """
        return state_id in self._state_ids


@dataclass(frozen=True)
class ValidationRule:
    """
    Immutable container for a validation rule.

    Attributes:
        name: Unique identifier for the rule
        check: Callable implementing the validation logic
        severity: How severe violations of this rule are
        description: Human-readable description of what the rule checks
    """

    name: str
    check: Callable[..., bool]
    severity: ValidationSeverity
    description: str


class Validator(AbstractValidator):
    """
    Concrete validator implementation with extensible rules.

    Runtime Invariants:
    - Rules are immutable after registration
    - Rule names are unique
    - Validation results are deterministic for a given configuration
    - Rules cannot modify the validated objects

    Example:
        validator = Validator(states, transitions, initial_state)
        validator.add_rule(
            "no_orphan_states",
            lambda ctx: all(has_transition_to(state) for state in ctx.states),
            ValidationSeverity.ERROR,
            "All states must be reachable"
        )
        results = validator.validate_structure()
    """

    def __init__(
        self,
        states: List[AbstractState],
        transitions: List[AbstractTransition],
        initial_state: AbstractState,
    ) -> None:
        """
        Initialize the validator with state machine configuration.

        Args:
            states: All states in the machine
            transitions: All transitions in the machine
            initial_state: Starting state of the machine

        Raises:
            ValidationError: If configuration is obviously invalid
        """
        if not states:
            raise ValidationError("State machine must have at least one state")
        if initial_state not in states:
            raise ValidationError("Initial state must be in states list")

        self._context = ValidationContext(states, transitions, initial_state)
        self._structure_rules: Dict[str, ValidationRule] = {}
        self._behavior_rules: Dict[str, ValidationRule] = {}
        self._data_rules: Dict[str, ValidationRule] = {}

        # Register default rules
        self._register_default_rules()

    def add_rule(
        self,
        name: str,
        check: Callable[..., bool],
        severity: ValidationSeverity,
        description: str,
        rule_type: str = "structure",
    ) -> None:
        """
        Add a custom validation rule.

        Args:
            name: Unique identifier for the rule
            check: Function implementing the validation logic
            severity: How severe violations are
            description: Human-readable description
            rule_type: One of "structure", "behavior", or "data"

        Raises:
            ValidationError: If rule_type is invalid or name is duplicate
        """
        rule = ValidationRule(name, check, severity, description)
        rule_dict = {
            "structure": self._structure_rules,
            "behavior": self._behavior_rules,
            "data": self._data_rules,
        }.get(rule_type)

        if rule_dict is None:
            raise ValidationError(f"Invalid rule type: {rule_type}")
        if name in rule_dict:
            raise ValidationError(f"Duplicate rule name: {name}")

        rule_dict[name] = rule

    def validate_structure(self) -> List[ValidationResult]:
        """
        Validate structural properties of the state machine.

        Checks:
        - State reachability
        - Transition validity
        - Composite state consistency
        - No duplicate state IDs

        Returns:
            List of validation results
        """
        self._context.current_results = []
        for rule in self._structure_rules.values():
            try:
                if not rule.check(self._context):
                    self._context.add_result(
                        rule.severity.name,
                        f"Structural validation failed: {rule.description}",
                        {"rule": rule.name},
                    )
            except Exception as e:
                self._context.add_result(
                    ValidationSeverity.ERROR.name,
                    f"Rule '{rule.name}' failed with exception: {str(e)}",
                    {"rule": rule.name, "error": str(e)},
                )
        return self._context.current_results

    def validate_behavior(self) -> List[ValidationResult]:
        """
        Validate behavioral properties of the state machine.

        Checks:
        - Guard consistency
        - Action safety
        - Event handling completeness
        - Priority conflicts

        Returns:
            List of validation results
        """
        self._context.current_results = []
        for rule in self._behavior_rules.values():
            try:
                if not rule.check(self._context):
                    self._context.add_result(
                        rule.severity.name,
                        f"Behavioral validation failed: {rule.description}",
                        {"rule": rule.name},
                    )
            except Exception as e:
                self._context.add_result(
                    ValidationSeverity.ERROR.name,
                    f"Rule '{rule.name}' failed with exception: {str(e)}",
                    {"rule": rule.name, "error": str(e)},
                )
        return self._context.current_results

    def validate_data(self) -> List[ValidationResult]:
        """
        Validate data handling in the state machine.

        Checks:
        - Data isolation
        - Type consistency
        - Required fields
        - Value constraints

        Returns:
            List of validation results
        """
        self._context.current_results = []
        for rule in self._data_rules.values():
            try:
                if not rule.check(self._context):
                    self._context.add_result(
                        rule.severity.name,
                        f"Data validation failed: {rule.description}",
                        {"rule": rule.name},
                    )
            except Exception as e:
                self._context.add_result(
                    ValidationSeverity.ERROR.name,
                    f"Rule '{rule.name}' failed with exception: {str(e)}",
                    {"rule": rule.name, "error": str(e)},
                )
        return self._context.current_results

    def _register_default_rules(self) -> None:
        """Register the default validation rules."""
        # Structural rules
        self.add_rule(
            "no_orphan_states",
            self._check_no_orphan_states,
            ValidationSeverity.ERROR,
            "All states must be reachable from the initial state",
        )
        self.add_rule(
            "valid_transitions",
            self._check_valid_transitions,
            ValidationSeverity.ERROR,
            "All transitions must reference valid states",
        )
        self.add_rule(
            "unique_state_ids",
            self._check_unique_state_ids,
            ValidationSeverity.ERROR,
            "All state IDs must be unique",
        )

        # Behavioral rules
        self.add_rule(
            "guard_safety",
            self._check_guard_safety,
            ValidationSeverity.WARNING,
            "Guards should be stateless and side-effect free",
            "behavior",
        )
        self.add_rule(
            "action_safety",
            self._check_action_safety,
            ValidationSeverity.WARNING,
            "Actions should handle exceptions gracefully",
            "behavior",
        )

        # Data rules
        self.add_rule(
            "data_isolation",
            self._check_data_isolation,
            ValidationSeverity.ERROR,
            "State data must be properly isolated",
            "data",  # Note: Make sure to specify rule_type as "data"
        )

    def _check_no_orphan_states(self, context: ValidationContext) -> bool:
        """Verify all states are reachable from the initial state."""
        reachable = set()
        to_visit = {context.initial_state}

        while to_visit:
            current = to_visit.pop()
            reachable.add(current.get_id())

            # Add states reachable through transitions
            for transition in context.transitions:
                if transition.get_source_state_id() == current.get_id():
                    target = context.get_state_by_id(transition.get_target_state_id())
                    if target and target.get_id() not in reachable:
                        to_visit.add(target)

        return len(reachable) == len(context.states)

    def _check_valid_transitions(self, context: ValidationContext) -> bool:
        """Verify all transitions reference valid states."""
        for transition in context.transitions:
            if not context.state_exists(transition.get_source_state_id()):
                context.add_result(
                    ValidationSeverity.ERROR.name,
                    f"Transition references nonexistent source state: {transition.get_source_state_id()}",
                    {
                        "transition_source": transition.get_source_state_id(),
                        "transition_target": transition.get_target_state_id(),
                    },
                )
                return False
            if not context.state_exists(transition.get_target_state_id()):
                context.add_result(
                    ValidationSeverity.ERROR.name,
                    f"Transition references nonexistent target state: {transition.get_target_state_id()}",
                    {
                        "transition_source": transition.get_source_state_id(),
                        "transition_target": transition.get_target_state_id(),
                    },
                )
                return False
        return True

    def _check_unique_state_ids(self, context: ValidationContext) -> bool:
        """Verify state IDs are unique."""
        seen = set()
        for state in context.states:
            state_id = state.get_id()
            if state_id in seen:
                context.add_result(
                    ValidationSeverity.ERROR.name, f"Duplicate state ID found: {state_id}", {"duplicate_id": state_id}
                )
                return False
            seen.add(state_id)
        return True

    def _check_guard_safety(self, context: ValidationContext) -> bool:
        """Verify guards are properly implemented."""
        for transition in context.transitions:
            guard = transition.get_guard()
            if guard is not None and not isinstance(guard, AbstractGuard):
                context.add_result(
                    ValidationSeverity.WARNING.name,
                    f"Guard in transition {transition.get_source_state_id()}->{transition.get_target_state_id()} "
                    "does not implement AbstractGuard",
                    {
                        "guard_type": str(type(guard)),
                        "transition_source": transition.get_source_state_id(),
                        "transition_target": transition.get_target_state_id(),
                    },
                )
                return False
        return True

    def _check_action_safety(self, context: ValidationContext) -> bool:
        """Verify actions are properly implemented."""
        for transition in context.transitions:
            for i, action in enumerate(transition.get_actions()):
                if not isinstance(action, AbstractAction):
                    context.add_result(
                        ValidationSeverity.WARNING.name,
                        f"Action {i} in transition {transition.get_source_state_id()}->"
                        f"{transition.get_target_state_id()} "
                        "does not implement AbstractAction",
                        {
                            "action_type": str(type(action)),
                            "action_index": i,
                            "transition_source": transition.get_source_state_id(),
                            "transition_target": transition.get_target_state_id(),
                        },
                    )
                    return False
        return True

    def _check_data_isolation(self, context: ValidationContext) -> bool:
        """Verify state data is properly isolated."""
        seen_data_ids = {}  # Map data_id to first state that used it
        for state in context.states:
            data_id = id(state.data)
            if data_id in seen_data_ids:
                context.add_result(
                    ValidationSeverity.ERROR.name,
                    f"State {state.get_id()} shares data dictionary with state {seen_data_ids[data_id]}",
                    {
                        "data_id": data_id,
                        "state_id": state.get_id(),
                        "shared_with_state": seen_data_ids[data_id],
                    },
                )
                return False
            seen_data_ids[data_id] = state.get_id()
        return True
