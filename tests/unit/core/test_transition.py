import pytest
from gotstate import State, Transition
from gotstate.core.guard import Guard
from gotstate.core.action import Action
from gotstate.types.common import EventId, StateId


def test_transition_creation():
    """Test basic transition creation."""
    source = State("source")
    target = State("target")
    
    transition = Transition(source, target, "test_event")
    
    assert transition.source is source
    assert transition.target is target
    assert transition.event_id == EventId("test_event")
    assert transition.guard is None
    assert len(transition.actions) == 0
    assert transition.is_internal is False
    assert transition.is_external is True
    assert transition.is_completion is False


def test_internal_transition():
    """Test internal transition (no target)."""
    source = State("source")
    
    transition = Transition(source, None, "test_event")
    
    assert transition.source is source
    assert transition.target is None
    assert transition.event_id == EventId("test_event")
    assert transition.is_internal is True
    assert transition.is_external is False


def test_completion_transition():
    """Test completion transition (no event)."""
    source = State("source")
    target = State("target")
    
    transition = Transition(source, target)
    
    assert transition.source is source
    assert transition.target is target
    assert transition.event_id is None
    assert transition.is_completion is True


def test_transition_with_guard():
    """Test transition with guard condition."""
    source = State("source")
    target = State("target")
    
    # Define a simple guard
    def guard_condition(event_id, event_data):
        return event_data.get("allowed", False)
    
    guard = Guard("test_guard", guard_condition)
    transition = Transition(source, target, "test_event", guard=guard)
    
    # Test guard evaluation
    assert transition.can_trigger(EventId("test_event"), {"allowed": True}) is True
    assert transition.can_trigger(EventId("test_event"), {"allowed": False}) is False
    assert transition.can_trigger(EventId("wrong_event"), {"allowed": True}) is False


def test_transition_with_actions():
    """Test transition with actions."""
    source = State("source")
    target = State("target")
    
    # Track action execution
    action_log = []
    
    def action_func(event_id, event_data):
        action_log.append(f"action: {event_id}")
    
    action = Action("test_action", action_func)
    transition = Transition(source, target, "test_event", actions=[action])
    
    # Execute actions
    transition.execute_actions(EventId("test_event"), {})
    
    assert len(action_log) == 1
    assert action_log[0] == "action: test_event"
    
    # Add another action
    def another_action(event_id, event_data):
        action_log.append(f"another: {event_id}")
    
    transition.add_action(Action("another_action", another_action))
    
    # Execute actions again
    transition.execute_actions(EventId("test_event"), {})
    
    assert len(action_log) == 3
    assert action_log[1] == "action: test_event"
    assert action_log[2] == "another: test_event" 