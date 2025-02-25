import pytest
from gotstate import State
from gotstate.types.common import StateId, EventId


def test_state_creation():
    """Test basic state creation."""
    state = State("test_state")
    assert state.id == StateId("test_state")
    assert state.name == "test_state"
    assert state.parent is None
    assert len(state.children) == 0


def test_state_hierarchy():
    """Test state hierarchy with parent-child relationships."""
    parent = State("parent")
    child = State("child", parent=parent)
    
    assert child.parent is parent
    assert child in parent.children
    assert parent.is_composite is True
    assert child.is_composite is False


def test_state_ancestors():
    """Test retrieving state ancestors."""
    grandparent = State("grandparent")
    parent = State("parent", parent=grandparent)
    child = State("child", parent=parent)
    
    ancestors = child.ancestors
    assert len(ancestors) == 2
    assert ancestors[0] is parent
    assert ancestors[1] is grandparent


def test_state_path():
    """Test state path generation."""
    grandparent = State("grandparent")
    parent = State("parent", parent=grandparent)
    child = State("child", parent=parent)
    
    assert child.path == "grandparent/parent/child"
    assert parent.path == "grandparent/parent"
    assert grandparent.path == "grandparent"


def test_state_actions():
    """Test state action execution."""
    state = State("test_state")
    
    # Track action execution with a simple list
    action_log = []
    
    @state.on_entry
    def entry_action(event_id, event_data):
        action_log.append(f"entry: {event_id}")
    
    @state.on_exit
    def exit_action(event_id, event_data):
        action_log.append(f"exit: {event_id}")
    
    # Execute actions
    state.execute_entry_actions(EventId("test_event"), {})
    state.execute_exit_actions(EventId("test_event"), {})
    
    assert len(action_log) == 2
    assert action_log[0] == "entry: test_event"
    assert action_log[1] == "exit: test_event" 