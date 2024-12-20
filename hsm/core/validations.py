# hsm/core/validation.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.transitions import Transition

if TYPE_CHECKING:
    from hsm.core.state_machine import StateMachine


class Validator:
    """
    Performs construction-time and runtime validation of the state machine,
    ensuring states, transitions, and events conform to defined rules.
    """

    def __init__(self) -> None:
        """
        Initialize the validator, potentially loading default or custom rules.
        """
        self._rules_engine = _ValidationRulesEngine()

    def validate_state_machine(self, state_machine: 'StateMachine') -> None:
        """
        Check the machine's states and transitions for consistency.

        :param state_machine: The state machine to validate.
        :raises ValidationError: If validation fails.
        """
        self._rules_engine.validate_machine(state_machine)

    def validate_transition(self, transition: "Transition") -> None:
        """
        Check that a given transition is well-formed.

        :param transition: The transition to validate.
        :raises ValidationError: If validation fails.
        """
        self._rules_engine.validate_transition(transition)

    def validate_event(self, event: "Event") -> None:
        """
        Validate that an event is well-defined and usable.

        :param event: The event to validate.
        :raises ValidationError: If validation fails.
        """
        self._rules_engine.validate_event(event)


class _ValidationRulesEngine:
    """
    Internal engine applying a set of validation rules to states, transitions,
    and events. Centralizes validation logic for easier maintenance.
    """

    def __init__(self) -> None:
        """
        Initialize internal rule sets. For simplicity, we rely on _DefaultValidationRules.
        """
        self._default_rules = _DefaultValidationRules

    def validate_machine(self, machine: 'StateMachine') -> None:
        """
        Apply all machine-level validation rules.

        :param machine: The state machine to validate.
        :raises ValidationError: If a rule fails.
        """
        self._default_rules.validate_machine(machine)

    def validate_transition(self, transition: "Transition") -> None:
        """
        Apply transition-level validation rules.

        :param transition: The transition to validate.
        :raises ValidationError: If a rule fails.
        """
        self._default_rules.validate_transition(transition)

    def validate_event(self, event: "Event") -> None:
        """
        Apply event-level validation rules.

        :param event: The event to validate.
        :raises ValidationError: If a rule fails.
        """
        self._default_rules.validate_event(event)


class _DefaultValidationRules:
    """
    Provides built-in validation rules ensuring basic correctness of states,
    transitions, and events out of the box.
    """

    @staticmethod
    def validate_machine(machine: 'StateMachine') -> None:
        """
        Check for basic machine correctness. For simplicity:
        - Ensure machine has an initial state.
        - (Optionally) Check that transitions reference valid states.
        """
        if machine.current_state is None:
            raise ValidationError("StateMachine must have an initial state.")
        # If needed, check transitions source/target states exist
        # This would require access to transitions from machine's context.
        # For now, we assume machine is minimal.

    @staticmethod
    def validate_transition(transition: "Transition") -> None:
        """
        Check that transition source/target states exist and guards are callable.
        """
        if transition.source is None or transition.target is None:
            raise ValidationError("Transition must have a valid source and target state.")
        # Optionally, ensure guards are callable:
        for g in transition.guards or []:
            if not callable(g):
                raise ValidationError("Transition guards must be callable.")
        # Actions must be callable too:
        for a in transition.actions or []:
            if not callable(a):
                raise ValidationError("Transition actions must be callable.")

    @staticmethod
    def validate_event(event: "Event") -> None:
        """
        Check that event name is non-empty.
        """
        if not event.name:
            raise ValidationError("Event must have a name.")
