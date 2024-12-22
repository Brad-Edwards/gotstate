# tests/unit/runtime/test_graph.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

"""Unit tests for the graph-based state machine structure."""

import pytest

from hsm.core.events import Event
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.runtime.graph import StateGraph


def test_add_state():
    """Test adding states to the graph."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()
    child = State("child")

    graph.add_state(composite)
    graph.add_state(child, parent=composite)

    # Verify hierarchy
    assert child.parent == composite
    assert child in graph.get_children(composite)
    assert graph._parent_map[child] == composite


def test_add_regular_state_parent():
    """Test adding states with a regular state as parent."""
    graph = StateGraph()
    parent = State("parent")
    child = State("child")

    graph.add_state(parent)
    graph.add_state(child, parent=parent)

    # Regular states don't update the state's parent attribute
    assert child.parent is None
    # But the graph tracks the relationship internally
    assert graph._parent_map[child] == parent
    assert child in graph.get_children(parent)


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
    root = CompositeState("root")
    root._children = set()
    parent = CompositeState("parent")
    parent._children = set()
    child = State("child")

    graph.add_state(root)
    graph.add_state(parent, parent=root)
    graph.add_state(child, parent=parent)

    ancestors = graph.get_ancestors(child)
    assert ancestors == [parent, root]


def test_validate_cycle_detection():
    """Test that cycle detection prevents invalid hierarchies."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()
    child = State("child")

    # First add composite as parent of child
    graph.add_state(composite)
    graph.add_state(child, parent=composite)

    # Then try to re-parent composite under child, which is not allowed
    with pytest.raises(ValueError, match="Cannot re-parent state"):
        graph.add_state(composite, parent=child)

    # Also test the reverse direction
    graph2 = StateGraph()
    composite2 = CompositeState("composite2")
    composite2._children = set()
    child2 = State("child2")

    # First add child as parent of composite
    graph2.add_state(child2)
    graph2.add_state(composite2, parent=child2)

    # Then try to re-parent child under composite, which is not allowed
    with pytest.raises(ValueError, match="Cannot re-parent state"):
        graph2.add_state(child2, parent=composite2)


def test_validate_composite_state():
    """Test validation of composite states."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()
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
