# tests/unit/test_custom_actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest

from gotstate.core.events import Event
from gotstate.core.states import State
from gotstate.plugins.custom_actions import MyCustomAction
from gotstate.runtime.graph import StateGraph


def test_custom_action_init():
    """Test initialization of MyCustomAction."""

    def action_fn(event):
        pass

    action = MyCustomAction(action_fn)
    assert action.action_fn == action_fn


def test_custom_action_execute():
    """Test action execution with event."""
    event_received = None

    def action_fn(event):
        nonlocal event_received
        event_received = event

    action = MyCustomAction(action_fn)
    test_event = Event("TestEvent")
    action.execute(test_event)

    assert event_received == test_event


def test_custom_action_run_alias():
    """Test that run is an alias for execute."""
    event_received = None

    def action_fn(event):
        nonlocal event_received
        event_received = event

    action = MyCustomAction(action_fn)
    test_event = Event("TestEvent")
    action.run(test_event)

    assert event_received == test_event


def test_custom_action_with_state_data():
    """Test action can modify state data."""
    graph = StateGraph()
    state = State("TestState")
    graph.add_state(state)

    def action_fn(event):
        if event.name == "update":
            graph.set_state_data(state, "processed", True)
            count = graph.get_state_data(state).get("count", 0)
            graph.set_state_data(state, "count", count + 1)

    action = MyCustomAction(action_fn)

    # Execute action
    action.execute(Event("update"))

    # Verify state data was modified
    state_data = graph.get_state_data(state)
    assert state_data["processed"] is True
    assert state_data["count"] == 1


def test_custom_action_with_exception():
    """Test action behavior when action function raises an exception."""

    def action_fn(event):
        raise ValueError("Test error")

    action = MyCustomAction(action_fn)
    test_event = Event("TestEvent")

    with pytest.raises(ValueError, match="Test error"):
        action.execute(test_event)
