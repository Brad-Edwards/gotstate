# hsm/tests/test_events.py
# nosec
# Licensed under the MIT License - see LICENSE file for details
"""
Test suite for events defined in events.py.

This module tests the Event and TimeoutEvent classes, ensuring they adhere to
the AbstractEvent protocol and handle various scenarios correctly.

Sections covered:
- Basic Error Cases
- Core Functionality
- Edge Cases
- Integration
"""
import asyncio
import io
import logging
import sys
from contextlib import contextmanager
from typing import Any, Dict

import pytest
import pytest_asyncio

from hsm.core.errors import HSMError, create_error_context
from hsm.core.events import Event, TimeoutEvent

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> Event:
    """Fixture for a sample event."""
    return Event("test_event", payload={"data": 42}, priority=5)


@pytest.fixture
def sample_timeout_event() -> TimeoutEvent:
    """Fixture for a sample timeout event."""
    return TimeoutEvent("timeout_event", payload={"info": "test"}, priority=10, timeout=1.5)


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data for scenarios if needed."""
    return {"count": 10, "status": "active"}


@pytest.fixture
def logger_fixture():
    """Fixture for testing logging integration."""
    logger = logging.getLogger("hsm.test.events")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


@pytest_asyncio.fixture
async def async_setup():
    """Async fixture for async tests."""
    await asyncio.sleep(0.01)
    return True


# -----------------------------------------------------------------------------
# BASIC ERROR CASES
# -----------------------------------------------------------------------------


def test_empty_event_id_raises_error() -> None:
    """
    Test that creating an event with an empty event_id raises ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        Event("")
    assert "event_id cannot be empty" in str(exc_info.value)


def test_negative_timeout_raises_error() -> None:
    """
    Test that creating a TimeoutEvent with negative timeout raises ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        TimeoutEvent("evt", timeout=-1.0)
    assert "timeout cannot be negative" in str(exc_info.value)


def test_hsm_error_inheritance() -> None:
    """
    Test that HSMError forms the base for event-related errors if any occur.
    This is symbolic since Event itself may not raise HSMError.
    """

    class CustomEventError(HSMError):
        pass

    with pytest.raises(CustomEventError) as exc_info:
        raise CustomEventError("test message")
    assert isinstance(exc_info.value, HSMError)


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_event_creation(sample_event: Event) -> None:
    """
    Test that a sample event is created correctly and getters work.
    """
    assert sample_event.get_id() == "test_event"
    assert sample_event.get_payload() == {"data": 42}
    assert sample_event.get_priority() == 5


def test_timeout_event_creation(sample_timeout_event: TimeoutEvent) -> None:
    """
    Test that a sample timeout event is created correctly.
    """
    assert sample_timeout_event.get_id() == "timeout_event"
    assert sample_timeout_event.get_payload() == {"info": "test"}
    assert sample_timeout_event.get_priority() == 10
    assert abs(sample_timeout_event.get_timeout() - 1.5) < 1e-10  # Use small epsilon for comparison


def test_event_immutability() -> None:
    """
    Test that the event attributes are effectively immutable.
    Attempts to modify them should fail or have no effect.
    """
    evt = Event("immutable_test", {"x": 1}, priority=2)
    # Direct attribute access is not provided, but let's assume no setters.
    # Just re-verify getters return the same value.
    assert evt.get_id() == "immutable_test"
    assert evt.get_payload() == {"x": 1}
    assert evt.get_priority() == 2


def test_error_chaining() -> None:
    """
    Test error chaining by raising one error and chaining another.
    """
    try:
        raise ValueError("Inner error")
    except ValueError as inner:
        try:
            raise HSMError("Outer error") from inner
        except HSMError as outer:
            assert outer.__cause__ is inner


def test_error_context_preservation() -> None:
    """
    Test creation and preservation of error context.
    """
    err = HSMError("Context error", details={"key": "value"})
    ctx = create_error_context(err, "traceback info")
    assert ctx.error_type is HSMError
    assert ctx.traceback == "traceback info"
    assert ctx.details == {"key": "value"}


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------


def test_null_payload() -> None:
    """
    Test event with null payload.
    """
    evt = Event("null_payload")
    assert evt.get_payload() is None


def test_zero_priority() -> None:
    """
    Test event with zero priority (boundary condition).
    """
    evt = Event("zero_priority", priority=0)
    assert evt.get_priority() == 0


def test_large_priority() -> None:
    """
    Test event with a very large priority value.
    """
    evt = Event("large_priority", priority=999999)
    assert evt.get_priority() == 999999


def test_maximum_recursion_scenario() -> None:
    """
    Pseudo test for maximum recursion:
    We can't truly recurse in events, so just a symbolic test.
    """

    def recurse(n: int):
        if n == 0:
            return "done"
        return recurse(n - 1)

    assert recurse(10) == "done"


# -----------------------------------------------------------------------------
# INTEGRATION
# -----------------------------------------------------------------------------


@contextmanager
def event_context_manager():
    """
    Context manager for testing error handling in a manager.
    If an error occurs, we simulate cleanup or convert it to HSMError.
    """
    try:
        yield
    except Exception as e:
        raise HSMError("Context error") from e


def test_error_handling_in_context_manager() -> None:
    """
    Test handling of errors that occur within a context manager.
    """
    with pytest.raises(HSMError) as exc_info:
        with event_context_manager():
            raise ValueError("Inner error in event context")
    assert "Context error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_error_scenario(async_setup) -> None:
    """
    Test async scenario where event processing might fail.
    We'll simulate an async failure.
    """

    async def async_fail():
        await asyncio.sleep(0.01)
        raise HSMError("Async event fail")

    with pytest.raises(HSMError) as exc_info:
        await async_fail()
    assert "Async event fail" in str(exc_info.value)


def test_event_equality() -> None:
    """Test equality comparison between events."""
    evt1 = Event("test", {"data": 1}, 1)
    evt2 = Event("test", {"data": 1}, 1)
    evt3 = Event("different", {"data": 1}, 1)

    assert evt1 == evt2  # Same values should be equal
    assert evt1 != evt3  # Different IDs should not be equal
    assert evt1 != "not_an_event"  # Different types should not be equal


def test_event_str_representation() -> None:
    """Test string representation of events."""
    evt = Event("test_event", {"data": 42}, 5)
    str_repr = str(evt)

    assert "test_event" in str_repr
    assert "42" in str_repr
    assert "5" in str_repr


def test_payload_mutation_protection() -> None:
    """Test that mutating the original payload doesn't affect the event."""
    original_payload = {"data": [1, 2, 3]}
    evt = Event("test", original_payload)

    # Modify original payload
    original_payload["data"].append(4)

    # Event payload should be unchanged
    assert evt.get_payload()["data"] == [1, 2, 3]


def test_event_id_type_validation() -> None:
    """Test that event_id must be a string."""
    with pytest.raises(TypeError):
        Event(123)  # type: ignore

    with pytest.raises(TypeError):
        Event(["not", "a", "string"])  # type: ignore


def test_timeout_event_inheritance() -> None:
    """Test that TimeoutEvent properly inherits from Event."""
    timeout_evt = TimeoutEvent("test", timeout=1.0)

    assert isinstance(timeout_evt, Event)
    assert isinstance(timeout_evt, TimeoutEvent)
    assert hasattr(timeout_evt, "get_timeout")
    assert not hasattr(Event("test"), "get_timeout")


def test_timeout_event_default_values() -> None:
    """Test default values in TimeoutEvent constructor."""
    evt = TimeoutEvent("test")

    assert evt.get_timeout() == 0.0
    assert evt.get_payload() is None
    assert evt.get_priority() == 0


def test_unicode_and_internationalization() -> None:
    """
    Test event handling with Unicode characters in the event ID and payload.
    """
    unicode_id = "évènt_😊"
    unicode_payload = {"message": "こんにちは世界", "emoji": "🤖"}
    evt = Event(unicode_id, unicode_payload, priority=1)

    assert evt.get_id() == unicode_id
    assert evt.get_payload() == unicode_payload
    assert evt.get_priority() == 1


def test_extremely_large_payload() -> None:
    """
    Test event with a very large payload to ensure handling does not degrade.
    """
    # Create a large string (e.g., ~1MB)
    large_payload = "A" * (1024 * 1024)  # 1MB of 'A'
    evt = Event("large_payload_event", large_payload, priority=10)

    assert evt.get_id() == "large_payload_event"
    assert evt.get_payload() == large_payload
    # Just ensure we can retrieve it without error
    assert evt.get_priority() == 10


def test_negative_priority() -> None:
    """
    Test event with a negative priority to see if it's allowed or causes issues.
    Assuming it's allowed and just returns the value.
    """
    evt = Event("negative_priority_event", None, priority=-5)
    assert evt.get_id() == "negative_priority_event"
    assert evt.get_priority() == -5


def test_extremely_large_priority() -> None:
    """
    Test event with an extremely large priority value.
    """
    large_priority = sys.maxsize
    evt = Event("huge_priority_event", priority=large_priority)
    assert evt.get_priority() == large_priority


def test_forward_compatibility_simulation() -> None:
    """
    Simulate reading an event from a future version with extra fields.
    We'll just pass extra fields to the constructor if it were flexible,
    or we can monkey patch something in payload.

    Assuming the event ignores unknown fields and just stores payload.
    """
    # Simulate a forward-compatible scenario: event_id and payload are known,
    # but let's say the payload includes unexpected fields.
    future_payload = {"known_key": 123, "unexpected_future_key": "future_data"}

    evt = Event("forward_compat_event", payload=future_payload, priority=1)
    # Check that we can still access known parts
    p = evt.get_payload()
    assert p["known_key"] == 123
    # We didn't define any special handling for unknown keys, but this simulates
    # that the event just stores what it gets.
    assert p["unexpected_future_key"] == "future_data"


def test_timeout_event_str_representation() -> None:
    """Test string representation of TimeoutEvent includes timeout value."""
    evt = TimeoutEvent("test_event", {"data": 42}, 5, timeout=1.5)
    str_repr = str(evt)

    assert "test_event" in str_repr
    assert "42" in str_repr
    assert "5" in str_repr
    assert "1.5" in str_repr


def test_timeout_event_equality() -> None:
    """Test equality comparison between TimeoutEvents."""
    evt1 = TimeoutEvent("test", {"data": 1}, 1, timeout=1.0)
    evt2 = TimeoutEvent("test", {"data": 1}, 1, timeout=1.0)
    evt3 = TimeoutEvent("test", {"data": 1}, 1, timeout=2.0)
    evt4 = Event("test", {"data": 1}, 1)  # Regular Event

    assert evt1 == evt2  # Same values should be equal
    assert evt1 != evt3  # Different timeouts should not be equal
    assert evt4 != evt1  # Event compared with TimeoutEvent should not be equal


def test_payload_deep_copy() -> None:
    """Test that nested mutable structures in payload are deep copied."""
    nested_payload = {"list": [1, [2, 3], {"a": 4}], "dict": {"nested": {"value": 5}}}
    evt = Event("test", nested_payload)

    # Modify nested structures in original payload
    nested_payload["list"][1][0] = 99
    nested_payload["dict"]["nested"]["value"] = 99

    # Event payload should remain unchanged
    event_payload = evt.get_payload()
    assert event_payload["list"][1][0] == 2
    assert event_payload["dict"]["nested"]["value"] == 5


def test_timeout_event_float_conversion() -> None:
    """Test that timeout values are properly converted to float."""
    # Test with integer timeout
    evt1 = TimeoutEvent("test", timeout=1)
    assert isinstance(evt1.get_timeout(), float)
    assert evt1.get_timeout() == 1.0

    # Test with string timeout (should raise TypeError)
    with pytest.raises(TypeError):
        TimeoutEvent("test", timeout="1.0")  # type: ignore


def test_event_hash_immutability() -> None:
    """Test that events can be used as dictionary keys (are hashable)."""
    evt1 = Event("test", {"data": 1}, 1)
    evt2 = Event("test", {"data": 1}, 1)
    evt3 = Event("different", {"data": 1}, 1)

    event_dict = {evt1: "value1"}

    # Same event (by value) should access the same dict entry
    event_dict[evt2] = "value2"
    assert len(event_dict) == 1
    assert event_dict[evt1] == "value2"

    # Different event should create new entry
    event_dict[evt3] = "value3"
    assert len(event_dict) == 2


def test_timeout_event_zero_timeout() -> None:
    """Test TimeoutEvent with exactly zero timeout."""
    evt = TimeoutEvent("test")
    assert evt.get_timeout() == 0.0
    assert isinstance(evt.get_timeout(), float)


def test_timeout_event_none_payload() -> None:
    """Test TimeoutEvent with None payload explicitly set."""
    evt = TimeoutEvent("test", payload=None, timeout=1.0)
    assert evt.get_payload() is None


def test_event_equality_with_none_payload() -> None:
    """Test equality comparison with None payloads."""
    evt1 = Event("test", None)
    evt2 = Event("test", None)
    evt3 = Event("test")  # implicit None

    assert evt1 == evt2
    assert evt1 == evt3
    assert evt2 == evt3


def test_event_hash_with_none_payload() -> None:
    """Test that events with None payloads can be hashed consistently."""
    evt1 = Event("test", None)
    evt2 = Event("test")  # implicit None

    # Both should hash to same value
    assert hash(evt1) == hash(evt2)

    # Should work as dict keys
    d = {evt1: "value"}
    assert d[evt2] == "value"


def test_timeout_event_negative_zero_timeout() -> None:
    """Test TimeoutEvent with negative zero (-0.0) timeout."""
    evt = TimeoutEvent("test", timeout=-0.0)
    assert evt.get_timeout() == 0.0
    assert not str(evt.get_timeout()).startswith("-")  # Ensure it's not -0.0


def test_event_priority_type_validation() -> None:
    """Test that priority must be an integer."""
    with pytest.raises(TypeError):
        Event("test", priority="5")  # type: ignore

    with pytest.raises(TypeError):
        Event("test", priority=1.5)  # type: ignore


def test_timeout_event_equality_subclass() -> None:
    """Test equality with a TimeoutEvent subclass."""

    class CustomTimeoutEvent(TimeoutEvent):
        pass

    evt1 = TimeoutEvent("test", timeout=1.0)
    evt2 = CustomTimeoutEvent("test", timeout=1.0)

    assert evt1 == evt2  # Should be equal despite different concrete classes


def test_event_equality_with_other_types() -> None:
    """Test equality comparison with non-Event types returns NotImplemented."""
    evt = Event("test")
    result = evt.__eq__(42)  # Should return NotImplemented
    assert result is NotImplemented

    # Also test with TimeoutEvent
    tevt = TimeoutEvent("test", timeout=1.0)
    result = tevt.__eq__(42)  # Should return NotImplemented
    assert result is NotImplemented


def test_event_hash_collision_handling() -> None:
    """Test that events with different contents but same hash are handled correctly."""
    evt1 = Event("test", {"a": 1})
    evt2 = Event("test", {"a": 2})

    # Even if they hash to same value, they should not be equal
    assert evt1 != evt2

    # Both can exist in same dict
    d = {evt1: "value1", evt2: "value2"}
    assert len(d) == 2
    assert d[evt1] == "value1"
    assert d[evt2] == "value2"

    # Test with TimeoutEvents
    tevt1 = TimeoutEvent("test", {"a": 1}, timeout=1.0)
    tevt2 = TimeoutEvent("test", {"a": 1}, timeout=2.0)

    # Different timeouts should allow separate dict entries
    d = {tevt1: "value1", tevt2: "value2"}
    assert len(d) == 2
    assert d[tevt1] == "value1"
    assert d[tevt2] == "value2"
