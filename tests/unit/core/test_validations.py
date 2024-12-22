# tests/unit/test_validations.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock

import pytest

from hsm.core.states import CompositeState
from hsm.core.validations import ValidationError
from hsm.runtime.graph import StateGraph


def test_validator_calls():
    from hsm.core.validations import Validator

    v = Validator()
    # Mock methods or assume no-op by default
    # Without specific rules, we just ensure calls do not fail.
    # We can also mock the _ValidationRulesEngine if needed.
    machine = MagicMock()
    transition = MagicMock()
    event = MagicMock()

    v.validate_state_machine(machine)
    v.validate_transition(transition)
    v.validate_event(event)
    # If defaults pass, no assertion needed. If strict checks are desired, we'd verify exceptions.


def test_default_validation_rules():
    from hsm.core.validations import _DefaultValidationRules

    # Without actual conditions, just a smoke test:
    machine = MagicMock()
    transition = MagicMock()
    event = MagicMock()

    # No exceptions thrown = pass
    _DefaultValidationRules.validate_machine(machine)
    _DefaultValidationRules.validate_transition(transition)
    _DefaultValidationRules.validate_event(event)


def test_reparenting_prevention():
    """Test that re-parenting states is not allowed."""
    graph = StateGraph()
    state1 = CompositeState("State1")
    state1._children = set()
    state2 = CompositeState("State2")
    state2._children = set()

    # Add state1 first
    graph.add_state(state1)
    # Add state2 as child of state1
    graph.add_state(state2, parent=state1)

    # Try to re-parent state1 under state2, which is not allowed
    with pytest.raises(ValueError, match="Cannot re-parent state"):
        graph.add_state(state1, parent=state2)


def test_validation_rules():
    """Test basic validation rules."""
    from hsm.core.validations import Validator, _DefaultValidationRules

    # Test validator calls
    v = Validator()
    machine = MagicMock()
    transition = MagicMock()
    event = MagicMock()

    v.validate_state_machine(machine)
    v.validate_transition(transition)
    v.validate_event(event)

    # Test default validation rules
    _DefaultValidationRules.validate_machine(machine)
    _DefaultValidationRules.validate_transition(transition)
    _DefaultValidationRules.validate_event(event)
