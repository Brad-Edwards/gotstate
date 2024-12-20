# tests/unit/test_errors.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_error_hierarchy(error_classes):
    HSMError, StateNotFoundError, TransitionError, ValidationError = error_classes
    assert issubclass(StateNotFoundError, HSMError)
    assert issubclass(TransitionError, HSMError)
    assert issubclass(ValidationError, HSMError)


def test_exceptions_instantiation(error_classes):
    HSMError, StateNotFoundError, TransitionError, ValidationError = error_classes
    e = StateNotFoundError("Missing state")
    assert str(e) == "Missing state"
    e = TransitionError("Bad transition")
    assert str(e) == "Bad transition"
    e = ValidationError("Invalid config")
    assert str(e) == "Invalid config"
