# tests/unit/test_custom_guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest

from hsm.core.events import Event
from hsm.core.states import State
from hsm.plugins.custom_guards import MyCustomGuard


def test_custom_guard_init():
    """Test initialization of MyCustomGuard."""

    def condition_fn(event):
        return True

    guard = MyCustomGuard(condition_fn)
    assert guard.condition_fn == condition_fn


def test_custom_guard_check_true():
    """Test guard check returns True when condition is met."""
    event_received = None

    def condition_fn(event):
        nonlocal event_received
        event_received = event
        return True

    guard = MyCustomGuard(condition_fn)
    test_event = Event("TestEvent")
    result = guard.check(test_event)

    assert result is True
    assert event_received == test_event


def test_custom_guard_check_false():
    """Test guard check returns False when condition is not met."""

    def condition_fn(event):
        return False

    guard = MyCustomGuard(condition_fn)
    test_event = Event("TestEvent")
    result = guard.check(test_event)

    assert result is False


def test_custom_guard_with_state_data():
    """Test guard can access state data in condition."""
    state = State("TestState")
    state.data["value"] = 15

    def condition_fn(event):
        return state.data["value"] > 10 and event.name == "test"

    guard = MyCustomGuard(condition_fn)

    # Test guard evaluation with matching event
    assert guard.check(Event("test")) is True

    # Test with non-matching event
    assert guard.check(Event("wrong")) is False

    # Test with changed state data
    state.data["value"] = 5
    assert guard.check(Event("test")) is False


def test_custom_guard_with_exception():
    """Test guard behavior when condition function raises an exception."""

    def condition_fn(event):
        raise ValueError("Test error")

    guard = MyCustomGuard(condition_fn)
    test_event = Event("TestEvent")

    with pytest.raises(ValueError, match="Test error"):
        guard.check(test_event)
