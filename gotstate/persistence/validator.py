"""
State machine definition validation management.

Validates state machine definitions, ensures semantic consistency,
and verifies transition rules.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

import icontract

from gotstate.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Defines validation detail levels."""

    BASIC = auto()
    NORMAL = auto()
    STRICT = auto()
    COMPLETE = auto()


class ValidationScope(Enum):
    """Defines validation scope boundaries."""

    LOCAL = auto()
    CONNECTED = auto()
    REGIONAL = auto()
    GLOBAL = auto()


class ValidationResult:
    """Encapsulates validation results with errors and warnings."""

    def __init__(self) -> None:
        self._errors: List[str] = []
        self._warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self._errors) == 0

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def add_error(self, message: str) -> None:
        self._errors.append(message)

    def add_warning(self, message: str) -> None:
        self._warnings.append(message)


@icontract.invariant(lambda self: isinstance(self._level, ValidationLevel), "Validation level must be valid")
class Validator:
    """Validates state machine definitions and semantics.

    Class Invariants:
    1. Validation level must be a valid ValidationLevel
    """

    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL) -> None:
        self._level = level
        self._rules: List[Callable[[Any], Optional[str]]] = []

    @property
    def level(self) -> ValidationLevel:
        return self._level

    def add_rule(self, rule: Callable[[Any], Optional[str]]) -> None:
        """Add a validation rule. Returns error message or None if valid."""
        self._rules.append(rule)

    def validate(self, machine: Any) -> ValidationResult:
        """Validate a state machine definition."""
        result = ValidationResult()

        for rule in self._rules:
            try:
                error = rule(machine)
                if error is not None:
                    result.add_error(error)
            except Exception as e:
                result.add_error(f"Rule raised exception: {e}")

        if self._level in (ValidationLevel.STRICT, ValidationLevel.COMPLETE):
            self._validate_structure(machine, result)

        return result

    def _validate_structure(self, machine: Any, result: ValidationResult) -> None:
        """Run built-in structural validation checks."""
        from gotstate.core.machine import StateMachine

        if not isinstance(machine, StateMachine):
            result.add_error("Machine must be a StateMachine instance")
            return

        if not machine.states:
            result.add_error("Machine has no states")

        for transition in machine.transitions:
            if transition.source.name not in machine.states:
                result.add_error(f"Transition source '{transition.source.name}' not registered with machine")
            if transition.target is not None and transition.target.name not in machine.states:
                result.add_error(f"Transition target '{transition.target.name}' not registered with machine")
