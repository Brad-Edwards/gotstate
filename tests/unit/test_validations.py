# tests/unit/test_validations.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock


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
