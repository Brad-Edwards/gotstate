# tests/unit/runtime/test_graph.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

"""Unit tests for the graph-based state machine structure."""

import threading
import time

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


def test_state_data_thread_safety():
    """Test thread-safe access to state data."""
    graph = StateGraph()
    state = State("test")
    graph.add_state(state)

    def modify_data():
        for i in range(100):
            graph.set_state_data(state, f"key{i}", i)
            time.sleep(0.001)  # Force thread switching

    threads = [threading.Thread(target=modify_data) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify data consistency
    data = graph.get_state_data(state)
    assert len(data) == 100
    for i in range(100):
        assert data[f"key{i}"] == i


def test_resolve_active_state():
    """Test resolving active state with history."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()
    initial = State("initial")
    other = State("other")

    graph.add_state(composite)
    graph.add_state(initial, parent=composite)
    graph.add_state(other, parent=composite)
    graph.set_initial_state(composite, initial)

    # First resolution should return initial state
    assert graph.resolve_active_state(composite) == initial

    # Record history and verify resolution
    graph.record_history(composite, other)
    assert graph.resolve_active_state(composite) == other


def test_get_composite_ancestors():
    """Test getting composite ancestors."""
    graph = StateGraph()
    root = CompositeState("root")
    root._children = set()
    middle = State("middle")  # Regular state
    leaf = CompositeState("leaf")
    leaf._children = set()
    final = State("final")

    graph.add_state(root)
    graph.add_state(middle, parent=root)
    graph.add_state(leaf, parent=middle)
    graph.add_state(final, parent=leaf)

    # Should only return composite states in reverse order
    ancestors = graph.get_composite_ancestors(final)
    assert ancestors == [root, leaf]


def test_clear_and_get_history():
    """Test history clearing and retrieval."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(composite)
    graph.add_state(state1, parent=composite)
    graph.add_state(state2, parent=composite)

    # Record some history
    graph.record_history(composite, state1)
    assert graph.get_history_state(composite) == state1

    graph.record_history(composite, state2)
    assert graph.get_history_state(composite) == state2

    # Clear history
    graph.clear_history()
    assert graph.get_history_state(composite) is None


def test_get_root_states():
    """Test getting root states."""
    graph = StateGraph()
    root1 = State("root1")
    root2 = State("root2")
    child = State("child")

    graph.add_state(root1)
    graph.add_state(root2)
    graph.add_state(child, parent=root1)

    roots = graph.get_root_states()
    assert len(roots) == 2
    assert root1 in roots
    assert root2 in roots
    assert child not in roots


def test_invalid_transition_states():
    """Test adding transitions with invalid states."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(state1)
    # state2 not added to graph

    with pytest.raises(ValueError, match="Target state.*not in graph"):
        graph.add_transition(Transition(source=state1, target=state2))

    with pytest.raises(ValueError, match="Source state.*not in graph"):
        graph.add_transition(Transition(source=state2, target=state1))


def test_get_all_states():
    """Test getting all states in the graph."""
    graph = StateGraph()
    states = [State(f"state{i}") for i in range(3)]

    for state in states:
        graph.add_state(state)

    all_states = graph.get_all_states()
    assert len(all_states) == 3
    for state in states:
        assert state in all_states


def test_invalid_initial_state():
    """Test setting invalid initial state."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()
    other_composite = CompositeState("other")
    other_composite._children = set()
    state = State("state")

    graph.add_state(composite)
    graph.add_state(other_composite)
    graph.add_state(state, parent=other_composite)

    # Try to set initial state that's not a child
    with pytest.raises(ValueError, match="must be a child of"):
        graph.set_initial_state(composite, state)

    # Try to set initial state for non-existent composite
    non_existent = CompositeState("non_existent")
    non_existent._children = set()
    with pytest.raises(ValueError, match="not in graph"):
        graph.set_initial_state(non_existent, state)
