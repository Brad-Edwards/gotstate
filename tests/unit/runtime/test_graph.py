"""Unit tests for the graph-based state machine structure."""

import pytest

from hsm.core.events import Event
from hsm.core.runtime.graph import StateGraph
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition


def test_add_state():
    """Test adding states to the graph."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(state1)
    graph.add_state(state2, parent=state1)

    # Verify hierarchy
    assert state2.parent == state1
    assert state2 in graph.get_children(state1)


def test_add_transition():
    """Test adding transitions to the graph."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(state1)
    graph.add_state(state2)

    transition = Transition(source=state1, target=state2)
    graph.add_transition(transition)

    # Verify transition is stored
    transitions = graph.get_valid_transitions(state1, Event("test"))
    assert transition in transitions


def test_get_ancestors():
    """Test retrieving ancestor states."""
    graph = StateGraph()
    root = State("root")
    parent = State("parent")
    child = State("child")

    graph.add_state(root)
    graph.add_state(parent, parent=root)
    graph.add_state(child, parent=parent)

    ancestors = graph.get_ancestors(child)
    assert ancestors == [parent, root]


def test_validate_cycle_detection():
    """Test that validation detects cycles in the hierarchy."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(state1)
    graph.add_state(state2, parent=state1)
    # Create a cycle by making state1 a child of state2
    graph.add_state(state1, parent=state2)

    # Validate should detect the cycle
    errors = graph.validate()
    assert any("cycle" in error.lower() for error in errors), "Cycle detection failed"


def test_validate_composite_state():
    """Test validation of composite states."""
    graph = StateGraph()
    composite = CompositeState("composite")
    graph.add_state(composite)

    # Composite state with no children should fail validation
    errors = graph.validate()
    assert any("no children" in error.lower() for error in errors)


def test_multiple_transitions():
    """Test handling multiple transitions from the same state."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")
    state3 = State("state3")

    graph.add_state(state1)
    graph.add_state(state2)
    graph.add_state(state3)

    t1 = Transition(source=state1, target=state2)
    t2 = Transition(source=state1, target=state3)
    graph.add_transition(t1)
    graph.add_transition(t2)

    # Both transitions should be available
    transitions = graph.get_valid_transitions(state1, Event("test"))
    assert len(transitions) == 2
    assert t1 in transitions
    assert t2 in transitions
