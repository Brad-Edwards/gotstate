# hsm/tests/test_guards.py
# Copyright (c) 2024
# Licensed under the MIT License - see LICENSE file for details
"""
Test suite for guards defined in guards.py.

This module contains unit tests for BasicGuard and mock implementations of AbstractGuard.
Integration tests have been moved to test_guards_integration.py.
"""
import asyncio
import logging
import random
import sys
from typing import Any, Dict

import pytest

from hsm.core.errors import (
    ActionExecutionError,
    ConfigurationError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    create_error_context,
)
from hsm.core.guards import (
    AsyncConditionGuard,
    BasicGuard,
    ConditionGuard,
    GuardBase,
    KeyExistsGuard,
    LoggingGuard,
    NoOpGuard,
)
from hsm.interfaces.abc import AbstractGuard
from hsm.interfaces.protocols import Event

# -----------------------------------------------------------------------------
# TEST HELPERS AND CONSTANTS
# -----------------------------------------------------------------------------

# Common test values for priorities
PRIORITY_TEST_CASES = {
    "standard": [-100, -1, 0, 1, 100],
    "system_bounds": [
        sys.maxsize,
        -sys.maxsize - 1,
        sys.maxsize - 1,
        (-sys.maxsize - 1) + 1,
    ],
    "very_large": [
        2**1000,  # Extremely large number
        -(2**1000),  # Extremely large negative
        sys.maxsize * sys.maxsize,  # Product of max sizes
    ],
}

# Common test values for payloads
PAYLOAD_TEST_CASES = {
    "basic_types": [
        None,
        42,
        "string",
        ["list"],
        {"dict": "value"},
        set(),
        True,
        0.5,
    ],
    "empty_collections": [
        "",  # Empty string
        b"",  # Empty bytes
        [],  # Empty list
        (),  # Empty tuple
        set(),  # Empty set
        {},  # Empty dict
        frozenset(),  # Empty frozenset
        bytearray(),  # Empty bytearray
    ],
}

# Common test values for event IDs
EVENT_ID_TEST_CASES = {
    "special": [
        "hello世界",  # Unicode
        "!@#$%^&*()",  # Special characters
        "\\n\\t\\r",  # Escape sequences
        "🌟🎉✨",  # Emojis
        " " * 100,  # Lots of spaces
        "\u0000\u0001\u0002",  # Control characters
    ],
    "invalid": [None, 42, True, [], {}],
}


def create_test_event(event_id: str = "test", payload: Any = None, priority: int = 0) -> Event:
    """Helper function to create test events with default values."""
    return MockEvent(event_id, payload, priority)


def create_large_data(depth: int = 100, width: int = 10000) -> Dict[str, Any]:
    """Helper function to create large test data structures."""
    large_data = {"deep_nest": {}, "large_list": list(range(width)), "large_string": "x" * 1000000}
    current = large_data["deep_nest"]
    for i in range(depth):
        current[f"level_{i}"] = {}
        current = current[f"level_{i}"]
    return large_data


# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class MockEvent:
    """A mock event implementing Event protocol."""

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0):
        if not isinstance(event_id, str):
            raise TypeError("event_id must be a string")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")

        self._id = event_id
        self._payload = payload
        self._priority = priority

    def get_id(self) -> str:
        return self._id

    def get_payload(self) -> Any:
        return self._payload

    def get_priority(self) -> int:
        return self._priority


class StatefulGuard(AbstractGuard):
    """A guard that maintains internal state for testing state isolation."""

    def __init__(self, initial_state: int = 0):
        self.state = initial_state

    def check(self, event: Event, state_data: Any) -> bool:
        self.state += 1
        return self.state % 2 == 0


class MockGuard(AbstractGuard):
    """A guard that returns a fixed boolean result."""

    def __init__(self, result: bool):
        self.result = result

    def check(self, event: Event, state_data: Any) -> bool:
        if event is None:
            raise ValueError("Event cannot be None")
        return self.result


class ErrorGuard(AbstractGuard):
    """A guard that always raises GuardEvaluationError."""

    def check(self, event: Event, state_data: Any) -> bool:
        raise GuardEvaluationError("Guard failed", "ErrorGuard", state_data, event)


class ComplexErrorGuard(AbstractGuard):
    """A guard that raises a different error depending on state_data."""

    def check(self, event: Event, state_data: Any) -> bool:
        if not isinstance(state_data, dict):
            raise ActionExecutionError("Invalid state_data type", "ComplexErrorGuard", state_data, event)
        if "invalid_state" in state_data:
            raise InvalidStateError("Invalid state encountered", "BadState", "check")
        if "bad_config" in state_data:
            raise ConfigurationError("Bad config", "GuardComponent", {"issue": "missing_key"})
        return False


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> Event:
    """A sample event fixture."""
    return MockEvent("test_event", payload={"data": 42}, priority=5)


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data fixture."""
    return {"count": 10, "status": "active"}


# -----------------------------------------------------------------------------
# EVENT PROTOCOL TESTING
# -----------------------------------------------------------------------------


def test_event_protocol_basic():
    """Test basic Event protocol implementation."""
    event = MockEvent("test", {"key": "value"}, 1)
    assert event.get_id() == "test"
    assert event.get_payload() == {"key": "value"}
    assert event.get_priority() == 1


def test_event_protocol_empty_values():
    """Test Event protocol with empty/default values."""
    event = MockEvent("")  # Empty event_id
    assert event.get_id() == ""
    assert event.get_payload() is None  # Default payload
    assert event.get_priority() == 0  # Default priority


def test_event_protocol_payload_types():
    """Test Event protocol with various payload types."""
    payloads = [
        None,
        42,
        "string",
        ["list"],
        {"dict": "value"},
        set(),
        True,
        0.5,
    ]

    for payload in payloads:
        event = MockEvent("test", payload)
        assert event.get_payload() == payload


def test_event_protocol_priority_bounds():
    """Test Event protocol with various priority values."""
    priorities = [-100, -1, 0, 1, 100]

    for priority in priorities:
        event = MockEvent("test", None, priority)
        assert event.get_priority() == priority


def test_event_protocol_type_safety():
    """Test Event protocol type safety."""
    # Test invalid event_id types
    invalid_ids = [None, 42, True, [], {}]
    for invalid_id in invalid_ids:
        with pytest.raises(TypeError):
            MockEvent(invalid_id)  # type: ignore

    # Test invalid priority types
    invalid_priorities = [None, "1", 1.0, True]
    for invalid_priority in invalid_priorities:
        with pytest.raises(TypeError):
            MockEvent("test", priority=invalid_priority)  # type: ignore


def test_event_protocol_special_characters():
    """Test Event protocol with special characters and Unicode."""
    special_ids = [
        "hello世界",  # Unicode
        "!@#$%^&*()",  # Special characters
        "\\n\\t\\r",  # Escape sequences
        "🌟🎉✨",  # Emojis
        " " * 100,  # Lots of spaces
        "\u0000\u0001\u0002",  # Control characters
    ]

    for special_id in special_ids:
        event = MockEvent(special_id)
        assert event.get_id() == special_id


def test_event_protocol_priority_overflow():
    """Test Event protocol with priority values near system limits."""
    # Test values near system limits
    edge_cases = [
        sys.maxsize,
        -sys.maxsize - 1,
        sys.maxsize - 1,
        (-sys.maxsize - 1) + 1,
    ]

    for priority in edge_cases:
        event = MockEvent("test", priority=priority)
        assert event.get_priority() == priority

    # Test with very large integers
    very_large_ints = [
        2**1000,  # Extremely large number
        -(2**1000),  # Extremely large negative
        sys.maxsize * sys.maxsize,  # Product of max sizes
    ]

    for large_int in very_large_ints:
        event = MockEvent("test", priority=large_int)
        assert event.get_priority() == large_int
        assert isinstance(event.get_priority(), int)

    # Test with non-integer numeric types that look like integers
    invalid_priorities = [
        float("inf"),  # Infinity
        float("-inf"),  # Negative infinity
        float("nan"),  # Not a number
        2.5,  # Non-integer float
        complex(1, 0),  # Complex number with zero imaginary part
    ]

    for invalid_priority in invalid_priorities:
        with pytest.raises(TypeError):
            MockEvent("test", priority=invalid_priority)  # type: ignore


def test_event_protocol_large_values():
    """Test Event protocol with large values."""
    # Test very large event_id
    large_id = "x" * 1000000
    event = MockEvent(large_id)
    assert event.get_id() == large_id

    # Test very large payload
    large_payload = {"key": "x" * 1000000}
    event = MockEvent("test", large_payload)
    assert event.get_payload() == large_payload

    # Test priority bounds
    max_priority = sys.maxsize
    min_priority = -sys.maxsize - 1

    event = MockEvent("test", priority=max_priority)
    assert event.get_priority() == max_priority

    event = MockEvent("test", priority=min_priority)
    assert event.get_priority() == min_priority


def test_event_protocol_empty_payloads():
    """Test Event protocol with various empty payloads."""
    empty_payloads = [
        "",  # Empty string
        b"",  # Empty bytes
        [],  # Empty list
        (),  # Empty tuple
        set(),  # Empty set
        {},  # Empty dict
        frozenset(),  # Empty frozenset
        bytearray(),  # Empty bytearray
    ]

    for payload in empty_payloads:
        event = MockEvent("test", payload)
        assert event.get_payload() == payload
        assert not bool(event.get_payload())  # Should be falsy


def test_event_protocol_priority_values():
    """Test Event protocol with various priority values including bounds and large numbers."""
    # Test all priority cases
    for case_name, priorities in PRIORITY_TEST_CASES.items():
        for priority in priorities:
            event = create_test_event(priority=priority)
            assert event.get_priority() == priority
            assert isinstance(event.get_priority(), int)

    # Test invalid priority types
    invalid_priorities = [None, "1", 1.0, True, float("inf"), float("-inf"), float("nan"), complex(1, 0)]
    for invalid_priority in invalid_priorities:
        with pytest.raises(TypeError):
            create_test_event(priority=invalid_priority)  # type: ignore


def test_event_protocol_payload_handling():
    """Test Event protocol with various payload types including empty collections."""
    # Test all payload cases
    for case_name, payloads in PAYLOAD_TEST_CASES.items():
        for payload in payloads:
            event = create_test_event(payload=payload)
            assert event.get_payload() == payload
            if case_name == "empty_collections":
                assert not bool(event.get_payload())


def test_event_protocol_id_handling():
    """Test Event protocol with various event ID types including special characters."""
    # Test valid special cases
    for special_id in EVENT_ID_TEST_CASES["special"]:
        event = create_test_event(event_id=special_id)
        assert event.get_id() == special_id

    # Test invalid ID types
    for invalid_id in EVENT_ID_TEST_CASES["invalid"]:
        with pytest.raises(TypeError):
            create_test_event(event_id=invalid_id)  # type: ignore

    # Test large ID
    large_id = "x" * 1000000
    event = create_test_event(event_id=large_id)
    assert event.get_id() == large_id


# -----------------------------------------------------------------------------
# GUARD STATE AND REUSABILITY
# -----------------------------------------------------------------------------


def test_guard_state_isolation():
    """Test that guard state is properly isolated."""
    guard = StatefulGuard()
    event = MockEvent("test")

    # First check should return False (state=1)
    assert guard.check(event, {}) is False
    # Second check should return True (state=2)
    assert guard.check(event, {}) is True
    # Third check should return False (state=3)
    assert guard.check(event, {}) is False

    # New guard instance should start fresh
    new_guard = StatefulGuard()
    assert new_guard.check(event, {}) is False


def test_guard_reusability(sample_event: Event):
    """Test that guards can be reused safely."""
    guard = MockGuard(True)

    # Use the same guard multiple times with different state data
    for _ in range(1000):
        assert guard.check(sample_event, {}) is True

    # Use with different events
    events = [MockEvent(f"event_{i}") for i in range(100)]
    for event in events:
        assert guard.check(event, {}) is True


def test_guard_immutability():
    """Test that guard behavior remains consistent."""
    guard = MockGuard(True)
    event = MockEvent("test")

    # Guard should maintain its result
    initial_result = guard.check(event, {})
    for _ in range(100):
        assert guard.check(event, {}) == initial_result

    # State data changes shouldn't affect guard behavior
    state_variations = [
        {},
        {"key": "value"},
        {"nested": {"key": "value"}},
        {"list": list(range(100))},
    ]

    for state in state_variations:
        assert guard.check(event, state) == initial_result


def test_guard_recursive_safety():
    """Test guard behavior with recursive calls."""

    class RecursiveGuard(AbstractGuard):
        def __init__(self, max_depth: int):
            self.max_depth = max_depth
            self.current_depth = 0

        def check(self, event: Event, state_data: Any) -> bool:
            self.current_depth += 1
            if self.current_depth > self.max_depth:
                raise RecursionError("Maximum recursion depth exceeded")

            if self.current_depth < self.max_depth:
                # Make recursive call
                return self.check(event, state_data)
            return True

    guard = RecursiveGuard(max_depth=100)
    event = MockEvent("test")

    # Test normal recursive case
    guard = RecursiveGuard(max_depth=10)
    assert guard.check(event, {}) is True

    # Test recursive depth limit
    guard = RecursiveGuard(max_depth=1000)
    with pytest.raises(RecursionError):
        guard.check(event, {})


def test_guard_initialization():
    """Test guard initialization edge cases."""

    class InitErrorGuard(AbstractGuard):
        def __init__(self, should_fail: bool = False):
            if should_fail:
                raise ValueError("Guard initialization failed")
            self.initialized = True

        def check(self, event: Event, state_data: Any) -> bool:
            return self.initialized

    # Test successful initialization
    guard = InitErrorGuard()
    event = MockEvent("test")
    assert guard.check(event, {}) is True

    # Test failed initialization
    with pytest.raises(ValueError) as exc_info:
        InitErrorGuard(should_fail=True)
    assert "Guard initialization failed" in str(exc_info.value)


def test_guard_state_reset():
    """Test guard state reset functionality."""

    class ResettableGuard(AbstractGuard):
        def __init__(self):
            self.reset()

        def reset(self):
            self.call_count = 0

        def check(self, event: Event, state_data: Any) -> bool:
            self.call_count += 1
            return self.call_count > 1

    guard = ResettableGuard()
    event = MockEvent("test")

    # First call should return False
    assert guard.check(event, {}) is False
    # Second call should return True
    assert guard.check(event, {}) is True

    # Reset state
    guard.reset()
    # After reset, first call should be False again
    assert guard.check(event, {}) is False


# -----------------------------------------------------------------------------
# ENHANCED ERROR HANDLING
# -----------------------------------------------------------------------------


def test_error_message_content(sample_event: Event):
    """Test error message content and attributes."""
    guard = ErrorGuard()

    try:
        guard.check(sample_event, {})
    except GuardEvaluationError as e:
        # Test error message
        assert str(e) == "Guard failed"
        assert e.guard_name == "ErrorGuard"
        assert e.state_data == {}
        assert e.event == sample_event

        # Test error attributes preservation after str/repr
        str_error = str(e)
        repr_error = repr(e)
        assert "Guard failed" in str_error
        assert "GuardEvaluationError" in repr_error


def test_nested_error_scenarios(sample_event: Event):
    """Test complex nested error scenarios."""
    guard = ComplexErrorGuard()

    # Test nested error chain
    try:
        try:
            try:
                guard.check(sample_event, ["not", "a", "dict"])
            except ActionExecutionError as e1:
                raise ValueError("Nested error 1") from e1
        except ValueError as e2:
            raise RuntimeError("Nested error 2") from e2
    except RuntimeError as e3:
        # Verify error chain
        assert isinstance(e3.__cause__, ValueError)
        assert isinstance(e3.__cause__.__cause__, ActionExecutionError)

        # Verify original error info is preserved
        original = e3.__cause__.__cause__
        assert original.state_data == ["not", "a", "dict"]
        assert original.event == sample_event


def test_error_with_large_data(sample_event: Event):
    """Test error handling with large data structures."""
    guard = ComplexErrorGuard()

    # Create a large state data structure
    large_data = {"deep_nest": {}, "large_list": list(range(10000)), "large_string": "x" * 1000000}
    current = large_data["deep_nest"]
    for i in range(100):
        current[f"level_{i}"] = {}
        current = current[f"level_{i}"]

    try:
        guard.check(sample_event, large_data)
    except GuardEvaluationError as e:
        # Verify large data is handled properly
        assert e.state_data == large_data
        assert e.event == sample_event

        # Verify error context handles large data
        ctx = create_error_context(e, "traceback")
        assert ctx.error_type is type(e)
        assert ctx.state_data == large_data


# -----------------------------------------------------------------------------
# BASIC ERROR CASES
# -----------------------------------------------------------------------------


def test_basic_guard_not_implemented(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that calling check on BasicGuard raises NotImplementedError.
    """
    guard = BasicGuard()
    with pytest.raises(NotImplementedError) as exc_info:
        guard.check(sample_event, sample_state_data)
    assert "must be subclassed" in str(exc_info.value)


def test_guard_evaluation_error(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that an ErrorGuard raises GuardEvaluationError as expected.
    """
    guard = ErrorGuard()
    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(sample_event, sample_state_data)
    assert "Guard failed" in str(exc_info.value)


def test_invalid_state_error(sample_event: Event) -> None:
    """
    Test that ComplexErrorGuard raises InvalidStateError when invalid_state key is present.
    """
    guard = ComplexErrorGuard()
    with pytest.raises(InvalidStateError) as exc_info:
        guard.check(sample_event, {"invalid_state": True})
    assert "Invalid state encountered" in str(exc_info.value)


def test_invalid_state_data_type(sample_event: Event) -> None:
    """
    Test that ComplexErrorGuard raises ActionExecutionError when state_data is not a dict.
    """
    guard = ComplexErrorGuard()
    with pytest.raises(ActionExecutionError) as exc_info:
        guard.check(sample_event, ["not", "a", "dict"])
    assert "Invalid state_data type" in str(exc_info.value)


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_mock_guard_true(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that a mock guard returning True works as expected.
    """
    guard = MockGuard(True)
    result = guard.check(sample_event, sample_state_data)
    assert result is True


def test_mock_guard_false(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test that a mock guard returning False works as expected.
    """
    guard = MockGuard(False)
    result = guard.check(sample_event, sample_state_data)
    assert result is False


def test_guard_none_inputs() -> None:
    """
    Test guard behavior with None inputs.
    """
    guard = MockGuard(True)

    # Test None event
    with pytest.raises(ValueError) as exc_info:
        guard.check(None, {})
    assert "Event cannot be None" in str(exc_info.value)

    # Test None state_data
    event = MockEvent("test")
    result = guard.check(event, None)
    assert result is True  # state_data can be None as it's Any type


def test_error_chaining(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test error chaining by raising a ConfigurationError and then chaining another error.
    """
    guard = ComplexErrorGuard()
    try:
        guard.check(sample_event, {"bad_config": True})
    except ConfigurationError as ce:
        try:
            raise HSMError("Chained error") from ce
        except HSMError as chained:
            assert chained.__cause__ is ce


def test_error_context_preservation(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test creation and preservation of error context.
    """
    try:
        raise GuardEvaluationError("Guard fail", "TestGuard", sample_state_data, sample_event)
    except GuardEvaluationError as ge:
        ctx = create_error_context(ge, "traceback")
        assert ctx.error_type is type(ge)
        assert ctx.traceback == "traceback"
        assert "Guard fail" in ge.message


# -----------------------------------------------------------------------------
# DATA STRUCTURE HANDLING
# -----------------------------------------------------------------------------


def test_state_data_handling(sample_event: Event) -> None:
    """
    Test guard behavior with various state data structures.
    Combines all state data testing into a single comprehensive test.
    """
    guard = MockGuard(True)

    # Test basic cases
    basic_cases = [
        {},  # Empty dict
        {"key": "value"},  # Simple flat
        {"null": None},  # None value
        {"bool": True},  # Boolean
        {"number": 42.0},  # Float
        {"string": ""},  # Empty string
        {"list": []},  # Empty list
        {"set": set()},  # Empty set
    ]

    for case in basic_cases:
        assert guard.check(sample_event, case) is True

    # Test nested structures
    nested_data = {
        "level1": {"level2": {"level3": {"array": [1, 2, {"deep": "value"}], "dict": {"nested_key": "nested_val"}}}},
        "metadata": {"version": "1.0"},
    }
    assert guard.check(sample_event, nested_data) is True

    # Test large data structures
    large_data = create_large_data()
    assert guard.check(sample_event, large_data) is True

    # Test random data
    for _ in range(5):
        random_payload = {
            "rand_int": random.randint(0, 1000),
            "rand_str": "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10)),
            "rand_list": [random.random() for _ in range(5)],
            "rand_dict": {f"key_{i}": random.random() for i in range(3)},
        }
        assert guard.check(sample_event, random_payload) is True


# -----------------------------------------------------------------------------
# GUARD IMPLEMENTATION TESTS
# -----------------------------------------------------------------------------


def test_guard_base_name():
    """Test guard_name property returns correct class name."""

    class CustomGuard(GuardBase):
        def check(self, event: Event, state_data: Any) -> bool:
            return True

    guard = CustomGuard()
    assert guard.guard_name == "CustomGuard"


def test_guard_base_error_raising():
    """Test _raise_guard_error helper method."""

    class ErrorRaisingGuard(GuardBase):
        def check(self, event: Event, state_data: Any) -> bool:
            self._raise_guard_error("Test error", state_data, event)

    guard = ErrorRaisingGuard()
    event = MockEvent("test")

    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(event, {})
    assert "Test error" in str(exc_info.value)
    assert exc_info.value.guard_name == "ErrorRaisingGuard"


def test_noop_guard():
    """Test NoOpGuard always returns True."""
    guard = NoOpGuard()
    event = MockEvent("test")

    # Test with various state data
    assert guard.check(event, None) is True
    assert guard.check(event, {}) is True
    assert guard.check(event, {"data": 123}) is True
    assert guard.check(event, [1, 2, 3]) is True


def test_logging_guard(caplog):
    """Test LoggingGuard logs correctly and returns True."""
    guard = LoggingGuard("test.logger")
    event = MockEvent("test_event", payload={"data": 42})
    state_data = {"status": "active"}

    with caplog.at_level(logging.INFO):
        result = guard.check(event, state_data)

    assert result is True
    assert len(caplog.records) == 1
    assert "test_event" in caplog.records[0].message
    assert "status" in caplog.records[0].message


def test_key_exists_guard():
    """Test KeyExistsGuard validates required keys."""
    guard = KeyExistsGuard(["status", "count"])
    event = MockEvent("test")

    # Test valid state data
    assert guard.check(event, {"status": "ok", "count": 42}) is True

    # Test missing key
    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(event, {"status": "ok"})
    assert "Missing required key: count" in str(exc_info.value)

    # Test non-dict state data
    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(event, ["not", "a", "dict"])
    assert "state_data must be a dictionary" in str(exc_info.value)


def test_condition_guard():
    """Test ConditionGuard evaluates conditions correctly."""

    def check_positive(state_data: Any) -> bool:
        return state_data.get("value", 0) > 0

    guard = ConditionGuard(check_positive)
    event = MockEvent("test")

    # Test passing condition
    assert guard.check(event, {"value": 42}) is True

    # Test failing condition
    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(event, {"value": -1})
    assert "Condition failed" in str(exc_info.value)

    # Test condition raising exception
    def failing_condition(state_data: Any) -> bool:
        raise ValueError("Bad value")

    guard = ConditionGuard(failing_condition)
    with pytest.raises(GuardEvaluationError) as exc_info:
        guard.check(event, {})
    assert "Condition evaluation error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_condition_guard():
    """Test AsyncConditionGuard evaluates async conditions correctly."""

    async def check_ready(state_data: Any) -> bool:
        await asyncio.sleep(0.01)
        return state_data.get("ready", False)

    guard = AsyncConditionGuard(check_ready)
    event = MockEvent("test")

    # Test passing condition
    assert await guard.check(event, {"ready": True}) is True

    # Test failing condition
    with pytest.raises(GuardEvaluationError) as exc_info:
        await guard.check(event, {"ready": False})
    assert "Async condition failed" in str(exc_info.value)

    # Test async condition raising exception
    async def failing_condition(state_data: Any) -> bool:
        raise ValueError("Async error")

    guard = AsyncConditionGuard(failing_condition)
    with pytest.raises(GuardEvaluationError) as exc_info:
        await guard.check(event, {})
    assert "Async condition evaluation error" in str(exc_info.value)
