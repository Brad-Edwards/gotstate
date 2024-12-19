import logging
from typing import Any, Dict

import pytest

from hsm.core.actions import BasicAction, LoggingAction, NoOpAction, SetDataAction, ValidateDataAction
from hsm.core.errors import ActionExecutionError
from hsm.interfaces.protocols import Event
from hsm.tests.utils import MockDataStructures


# Mock Event class for testing
class MockEvent:
    def __init__(self, event_id: str):
        self._id = event_id

    def get_id(self) -> str:
        return self._id


# Common test fixtures that can be shared across test classes
@pytest.fixture
def mock_event():
    return MockEvent("test_event")


@pytest.fixture
def state_data() -> Dict[str, Any]:
    return {"key1": "value1", "counter": 5}


@pytest.fixture
def original_state_data(state_data) -> Dict[str, Any]:
    return state_data.copy()


# Helper functions for common test patterns
def assert_state_unchanged(state_data: Dict[str, Any], original_data: Dict[str, Any]):
    """Assert that state_data hasn't been modified"""
    assert state_data == original_data


def assert_action_preserves_state(action: BasicAction, mock_event: Event, state_data: Dict[str, Any]):
    """Assert that an action doesn't modify state"""
    original_data = state_data.copy()
    action.execute(mock_event, state_data)
    assert_state_unchanged(state_data, original_data)


class ActionTestBase:
    """Base class for common action test functionality"""

    def assert_raises_with_none_event(self, action: BasicAction, state_data: Dict[str, Any]):
        with pytest.raises(AttributeError):
            action.execute(None, state_data)

    def assert_raises_with_none_state(self, action: BasicAction, mock_event: Event):
        with pytest.raises(NotImplementedError):
            action.execute(mock_event, None)


# BasicAction Tests
class TestBasicAction:
    def test_basic_action_raises_not_implemented(self, mock_event, state_data):
        action = BasicAction()
        with pytest.raises(NotImplementedError) as exc_info:
            action.execute(mock_event, state_data)
        assert "BasicAction must be subclassed" in str(exc_info.value)

    def test_basic_action_with_none_event(self, state_data):
        action = BasicAction()
        with pytest.raises(NotImplementedError):
            action.execute(None, state_data)

    def test_basic_action_with_none_state(self, mock_event):
        action = BasicAction()
        with pytest.raises(NotImplementedError):
            action.execute(mock_event, None)


# NoOpAction Tests
class TestNoOpAction:
    def test_noop_action_does_nothing(self, mock_event, state_data):
        action = NoOpAction()
        original_data = state_data.copy()
        action.execute(mock_event, state_data)
        assert state_data == original_data

    def test_noop_action_with_none_event(self, state_data):
        action = NoOpAction()
        original_data = state_data.copy()
        action.execute(None, state_data)
        assert state_data == original_data

    def test_noop_action_with_none_state(self, mock_event):
        action = NoOpAction()
        action.execute(mock_event, None)  # Should not raise

    def test_noop_action_with_empty_dict(self, mock_event):
        action = NoOpAction()
        state_data = {}
        action.execute(mock_event, state_data)
        assert state_data == {}


# LoggingAction Tests
class TestLoggingAction:
    @pytest.mark.parametrize(
        "log_level,should_log", [(logging.DEBUG, True), (logging.INFO, True), (logging.ERROR, False)]
    )
    def test_logging_action_levels(self, mock_event, state_data, caplog, log_level, should_log):
        action = LoggingAction()
        caplog.set_level(log_level)
        action.execute(mock_event, state_data)
        log_contains_event = "test_event" in caplog.text
        assert log_contains_event == should_log

    def test_logging_action_custom_logger(self, mock_event, state_data):
        custom_logger = "custom.logger"
        action = LoggingAction(logger_name=custom_logger)
        assert action.logger.name == custom_logger

    def test_logging_action_preserves_state(self, mock_event, state_data):
        action = LoggingAction()
        original_data = state_data.copy()
        action.execute(mock_event, state_data)
        assert state_data == original_data

    def test_logging_action_with_none_event(self, state_data, caplog):
        action = LoggingAction()
        caplog.set_level(logging.INFO)
        with pytest.raises(AttributeError):
            action.execute(None, state_data)

    def test_logging_action_with_none_state(self, mock_event, caplog):
        action = LoggingAction()
        caplog.set_level(logging.INFO)
        action.execute(mock_event, None)
        assert "test_event" in caplog.text
        assert "None" in caplog.text

    def test_logging_action_with_empty_dict(self, mock_event, caplog):
        action = LoggingAction()
        caplog.set_level(logging.INFO)
        action.execute(mock_event, {})
        assert "test_event" in caplog.text
        assert "{}" in caplog.text

    def test_logging_action_with_malformed_event(self, state_data, caplog):
        class MalformedEvent:
            # Missing get_id method required by Event protocol
            pass

        action = LoggingAction()
        caplog.set_level(logging.INFO)
        with pytest.raises(AttributeError):
            action.execute(MalformedEvent(), state_data)

    def test_logging_action_with_failing_logger(self, mock_event, state_data):
        class FailingLogger:
            def info(self, msg):
                raise RuntimeError("Logger failure")

        action = LoggingAction()
        action.logger = FailingLogger()
        with pytest.raises(RuntimeError):
            action.execute(mock_event, state_data)


# SetDataAction Tests
class TestSetDataAction:
    def test_set_data_action_sets_value(self, mock_event, state_data):
        action = SetDataAction("new_key", "new_value")
        action.execute(mock_event, state_data)
        assert state_data.get("new_key") == "new_value"

    def test_set_data_action_overwrites_existing(self, mock_event, state_data):
        action = SetDataAction("key1", "new_value")
        action.execute(mock_event, state_data)
        assert state_data.get("key1") == "new_value"

    def test_set_data_action_invalid_state_data(self, mock_event):
        action = SetDataAction("key", "value")
        invalid_state = "not_a_dict"

        with pytest.raises(ActionExecutionError) as exc_info:
            action.execute(mock_event, invalid_state)
        assert "State data must be a dictionary" in str(exc_info.value)

    def test_set_data_action_rollback_on_error(self, mock_event, state_data):
        class ErrorInContextAction(SetDataAction):
            def execute(self, event: Event, state_data: Any) -> None:
                with self._temporary_change(state_data):
                    raise RuntimeError("Simulated error")

        action = ErrorInContextAction("test_key", "test_value")
        original_data = state_data.copy()

        with pytest.raises(RuntimeError):
            action.execute(mock_event, state_data)

        assert state_data == original_data

    def test_set_data_action_rollback_preserves_none(self, mock_event, state_data):
        action = SetDataAction("new_key", "new_value")

        # Simulate an error after setting the value
        class SimulatedError(Exception):
            pass

        with pytest.raises(SimulatedError):
            with action._temporary_change(state_data):
                raise SimulatedError()

        assert "new_key" not in state_data

    def test_set_data_action_with_none_key(self, mock_event, state_data):
        action = SetDataAction("key", None)  # None value is allowed
        action.execute(mock_event, state_data)
        assert state_data.get("key") is None

    def test_set_data_action_with_empty_key(self, mock_event, state_data):
        action = SetDataAction("", "value")
        action.execute(mock_event, state_data)
        assert state_data.get("") == "value"

    def test_set_data_action_with_none_value(self, mock_event, state_data):
        action = SetDataAction("key", None)
        action.execute(mock_event, state_data)
        assert state_data.get("key") is None

    def test_set_data_action_with_nested_dict(self, mock_event):
        state_data = {"nested": {"key": "value"}}
        action = SetDataAction("nested", {"new_key": "new_value"})
        action.execute(mock_event, state_data)
        assert state_data["nested"] == {"new_key": "new_value"}

    def test_set_data_action_with_list_value(self, mock_event, state_data):
        action = SetDataAction("list_key", [1, 2, 3])
        action.execute(mock_event, state_data)
        assert state_data.get("list_key") == [1, 2, 3]

    def test_set_data_action_with_non_string_key(self, mock_event, state_data):
        with pytest.raises(TypeError, match="key must be a string"):
            action = SetDataAction(123, "value")  # numeric key should raise TypeError

    def test_set_data_action_with_bool_key(self, mock_event, state_data):
        with pytest.raises(TypeError, match="key must be a string"):
            action = SetDataAction(True, "value")

    def test_set_data_action_with_float_key(self, mock_event, state_data):
        with pytest.raises(TypeError, match="key must be a string"):
            action = SetDataAction(1.23, "value")

    def test_set_data_action_with_none_as_key(self, mock_event, state_data):
        with pytest.raises(TypeError, match="key must be a string"):
            action = SetDataAction(None, "value")

    def test_set_data_action_with_object_as_key(self, mock_event, state_data):
        class CustomObject:
            def __str__(self):
                return "custom_string"

        with pytest.raises(TypeError, match="key must be a string"):
            action = SetDataAction(CustomObject(), "value")

    def test_set_data_action_with_very_large_value(self, mock_event, state_data):
        """Test setting a very large value to ensure no memory/performance issues"""
        large_value = MockDataStructures.create_large_dict(1000)
        action = SetDataAction("large_key", large_value)
        action.execute(mock_event, state_data)
        assert state_data["large_key"] == large_value


# ValidateDataAction Tests
class TestValidateDataAction:
    @pytest.fixture
    def condition(self):
        return lambda data: data.get("counter") > 0

    def test_validate_data_action_passes(self, mock_event, state_data, condition):
        action = ValidateDataAction(["counter"], condition)
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_missing_key(self, mock_event, condition):
        action = ValidateDataAction(["missing_key"], condition)
        invalid_data = {"other_key": "value"}

        with pytest.raises(ActionExecutionError) as exc_info:
            action.execute(mock_event, invalid_data)
        assert "Missing required key" in str(exc_info.value)

    def test_validate_data_action_condition_fails(self, mock_event, condition):
        action = ValidateDataAction(["counter"], condition)
        invalid_data = {"counter": 0}  # Will fail condition > 0

        with pytest.raises(ActionExecutionError) as exc_info:
            action.execute(mock_event, invalid_data)
        assert "Validation condition failed" in str(exc_info.value)

    def test_validate_data_action_invalid_state_data(self, mock_event, condition):
        action = ValidateDataAction(["key"], condition)
        invalid_state = "not_a_dict"

        with pytest.raises(ActionExecutionError) as exc_info:
            action.execute(mock_event, invalid_state)
        assert "Invalid state_data type" in str(exc_info.value)

    def test_validate_data_action_multiple_keys(self, mock_event, state_data):
        action = ValidateDataAction(["key1", "counter"], lambda data: all(k in data for k in ["key1", "counter"]))
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_empty_required_keys(self, mock_event, state_data):
        action = ValidateDataAction([], lambda data: True)
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_preserves_state(self, mock_event, state_data, condition):
        action = ValidateDataAction(["counter"], condition)
        original_data = state_data.copy()
        action.execute(mock_event, state_data)
        assert state_data == original_data

    def test_validate_data_action_with_none_required_keys(self, mock_event, state_data):
        with pytest.raises(ValueError):  # Change to ValueError
            ValidateDataAction(None, lambda x: True)

    def test_validate_data_action_with_none_condition(self, mock_event, state_data):
        with pytest.raises(ValueError):  # Change to ValueError
            ValidateDataAction(["key"], None)

    def test_validate_data_action_with_empty_string_key(self, mock_event, state_data):
        action = ValidateDataAction([""], lambda x: True)
        state_data.setdefault("", "value")
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_complex_structures(self, mock_event):
        test_cases = [
            {
                "data": {"nested": {"key": "value"}},
                "keys": ["nested"],
                "condition": lambda data: isinstance(data.get("nested"), dict) and "key" in data.get("nested", {}),
            },
            {
                "data": {"counter": 1, "key1": "value1"},
                "keys": ["counter", "key1"],
                "condition": lambda data: (
                    data.get("counter", 0) > 0 and isinstance(data.get("key1"), str) and len(data.get("key1", "")) > 0
                ),
            },
        ]

        for case in test_cases:
            action = ValidateDataAction(case["keys"], case["condition"])
            action.execute(mock_event, case["data"])  # Should not raise

    def test_validate_data_action_with_list_values(self, mock_event):
        state_data = {"list": [1, 2, 3]}
        action = ValidateDataAction(
            ["list"], lambda data: isinstance(data.get("list"), list) and len(data.get("list", [])) > 0
        )
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_with_duplicate_keys(self, mock_event, state_data):
        action = ValidateDataAction(["counter", "counter"], lambda x: True)
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_with_special_characters(self, mock_event):
        state_data = {"@#$%^": "value"}
        action = ValidateDataAction(["@#$%^"], lambda x: True)
        action.execute(mock_event, state_data)  # Should not raise

    def test_validate_data_action_with_modifying_condition(self, mock_event):
        def bad_condition(data):
            data["new_key"] = "modified"  # Condition shouldn't modify data
            return True

        action = ValidateDataAction(["key1"], bad_condition)
        state_data = {"key1": "value1"}
        original_data = state_data.copy()

        action.execute(mock_event, state_data)
        assert state_data == original_data, "Condition should not modify state_data"

    def test_validate_data_action_with_recursive_validation(self, mock_event):
        def recursive_condition(data):
            if "nested" not in data:
                return False
            nested = data["nested"]
            return isinstance(nested, dict) and "required" in nested and nested["required"] == "value"

        action = ValidateDataAction(["nested"], recursive_condition)

        # Valid nested structure
        valid_data = {"nested": {"required": "value"}}
        action.execute(mock_event, valid_data)  # Should not raise

        # Invalid nested structure
        invalid_data = {"nested": {"wrong_key": "value"}}
        with pytest.raises(ActionExecutionError):
            action.execute(mock_event, invalid_data)

    def test_validate_data_action_with_invalid_condition_type(self, mock_event):
        # Test with non-callable condition
        with pytest.raises(TypeError):
            ValidateDataAction(["key"], "not_a_function")

    def test_validate_data_action_with_invalid_required_keys_type(self, mock_event):
        # Test with non-list required_keys
        with pytest.raises(TypeError):
            ValidateDataAction("not_a_list", lambda x: True)

    def test_validate_data_action_condition_exception(self, mock_event):
        def failing_condition(data):
            raise ValueError("Condition failed unexpectedly")

        action = ValidateDataAction(["key"], failing_condition)
        with pytest.raises(ActionExecutionError):
            action.execute(mock_event, {"key": "value"})

    def test_validate_data_action_with_large_nested_structure(self, mock_event):
        deep_dict = MockDataStructures.create_deep_dict(50)
        large_dict = MockDataStructures.create_large_dict(100)

        state_data = {"deep": deep_dict, "large": large_dict}

        def validate_structure(data):
            return (
                isinstance(data.get("deep"), dict)
                and isinstance(data.get("large"), dict)
                and len(data.get("large", {})) == 100
            )

        action = ValidateDataAction(["deep", "large"], validate_structure)
        action.execute(mock_event, state_data)  # Should not raise


class TestSetDataActionIntegration:
    def test_set_data_action_rollback_on_error(self, mock_event, state_data):
        class ErrorInContextAction(SetDataAction):
            def execute(self, event: Event, state_data: Any) -> None:
                with self._temporary_change(state_data):
                    raise RuntimeError("Simulated error")

        action = ErrorInContextAction("test_key", "test_value")
        original_data = state_data.copy()

        with pytest.raises(RuntimeError):
            action.execute(mock_event, state_data)

        assert state_data == original_data


class TestValidateDataActionIntegration:
    def test_validate_data_action_with_large_nested_structure(self, mock_event):
        # Move complex nested structure tests here
        pass
