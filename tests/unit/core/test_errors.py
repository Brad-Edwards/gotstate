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


def test_hsm_error_base():
    from hsm.core.errors import HSMError

    error = HSMError("Base error")
    assert str(error) == "Base error"
    assert isinstance(error, Exception)


def test_error_inheritance():
    from hsm.core.errors import HSMError, StateNotFoundError, TransitionError, ValidationError

    # Test that each error is both its type and HSMError
    state_error = StateNotFoundError("state")
    assert isinstance(state_error, StateNotFoundError)
    assert isinstance(state_error, HSMError)

    trans_error = TransitionError("transition")
    assert isinstance(trans_error, TransitionError)
    assert isinstance(trans_error, HSMError)

    val_error = ValidationError("validation")
    assert isinstance(val_error, ValidationError)
    assert isinstance(val_error, HSMError)


def test_error_messages():
    from hsm.core.errors import HSMError, StateNotFoundError, TransitionError, ValidationError

    # Test custom error messages
    assert str(HSMError("Custom base")) == "Custom base"
    assert str(StateNotFoundError("Custom state")) == "Custom state"
    assert str(TransitionError("Custom transition")) == "Custom transition"
    assert str(ValidationError("Custom validation")) == "Custom validation"


def test_error_empty_messages():
    from hsm.core.errors import HSMError, StateNotFoundError, TransitionError, ValidationError

    # Test empty error messages
    assert str(HSMError()) == ""
    assert str(StateNotFoundError()) == ""
    assert str(TransitionError()) == ""
    assert str(ValidationError()) == ""
