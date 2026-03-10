"""Tests for gotstate.persistence modules (serializer, validator)."""

import json

import pytest

from gotstate.core.machine import StateMachine
from gotstate.core.state import State
from gotstate.core.transition import Transition
from gotstate.exceptions import SerializationError, ValidationError
from gotstate.persistence.serializer import Serializer, SerializationFormat
from gotstate.persistence.validator import (
    ValidationLevel,
    ValidationResult,
    Validator,
)


class TestSerializer:
    def _make_machine(self):
        m = StateMachine("test")
        s1, s2 = State("idle"), State("active")
        m.add_state(s1)
        m.add_state(s2)
        m.set_initial_state(s1)
        m.add_transition(Transition(s1, s2, trigger="go"))
        return m

    def test_create(self):
        s = Serializer()
        assert s.format == SerializationFormat.JSON
        assert s.version == "1.0.0"

    def test_serialize_json(self):
        m = self._make_machine()
        m.start()
        s = Serializer()
        result = s.serialize(m)
        data = json.loads(result)
        assert data["name"] == "test"
        assert data["status"] == "ACTIVE"
        assert len(data["states"]) == 2
        assert len(data["transitions"]) == 1
        assert data["current_state"] == "idle"

    def test_serialize_non_machine_raises(self):
        s = Serializer()
        with pytest.raises(SerializationError, match="StateMachine"):
            s.serialize("not a machine")

    def test_serialize_unsupported_format(self):
        s = Serializer(SerializationFormat.BINARY)
        m = self._make_machine()
        with pytest.raises(SerializationError, match="Unsupported"):
            s.serialize(m)

    def test_deserialize_json(self):
        s = Serializer()
        data = '{"name": "test", "states": []}'
        result = s.deserialize(data)
        assert result["name"] == "test"

    def test_deserialize_invalid_json(self):
        s = Serializer()
        with pytest.raises(SerializationError, match="Invalid JSON"):
            s.deserialize("{invalid")

    def test_deserialize_unsupported_format(self):
        s = Serializer(SerializationFormat.XML)
        with pytest.raises(SerializationError, match="Unsupported"):
            s.deserialize("<xml/>")


class TestValidationResult:
    def test_empty_is_valid(self):
        r = ValidationResult()
        assert r.is_valid
        assert r.errors == []
        assert r.warnings == []

    def test_add_error_makes_invalid(self):
        r = ValidationResult()
        r.add_error("something wrong")
        assert not r.is_valid
        assert len(r.errors) == 1

    def test_warnings_dont_affect_validity(self):
        r = ValidationResult()
        r.add_warning("minor issue")
        assert r.is_valid
        assert len(r.warnings) == 1


class TestValidator:
    def test_create(self):
        v = Validator()
        assert v.level == ValidationLevel.NORMAL

    def test_custom_rule_pass(self):
        v = Validator()
        v.add_rule(lambda m: None)  # Always passes
        result = v.validate("anything")
        assert result.is_valid

    def test_custom_rule_fail(self):
        v = Validator()
        v.add_rule(lambda m: "bad thing happened")
        result = v.validate("anything")
        assert not result.is_valid
        assert "bad thing happened" in result.errors

    def test_rule_exception_captured(self):
        v = Validator()
        v.add_rule(lambda m: (_ for _ in ()).throw(RuntimeError("oops")))
        result = v.validate("anything")
        assert not result.is_valid
        assert any("exception" in e for e in result.errors)

    def test_strict_validation_no_states(self):
        v = Validator(ValidationLevel.STRICT)
        m = StateMachine("empty")
        result = v.validate(m)
        assert not result.is_valid
        assert any("no states" in e for e in result.errors)

    def test_strict_validation_valid_machine(self):
        v = Validator(ValidationLevel.STRICT)
        m = StateMachine("test")
        s1, s2 = State("s1"), State("s2")
        m.add_state(s1)
        m.add_state(s2)
        m.add_transition(Transition(s1, s2, trigger="go"))
        result = v.validate(m)
        assert result.is_valid

    def test_strict_validation_unregistered_transition_source(self):
        v = Validator(ValidationLevel.STRICT)
        m = StateMachine("test")
        s1 = State("s1")
        m.add_state(s1)
        foreign = State("foreign")
        m.add_transition(Transition(foreign, s1, trigger="go"))
        result = v.validate(m)
        assert not result.is_valid
        assert any("source" in e and "foreign" in e for e in result.errors)

    def test_non_machine_strict_validation(self):
        v = Validator(ValidationLevel.STRICT)
        result = v.validate("not a machine")
        assert not result.is_valid
