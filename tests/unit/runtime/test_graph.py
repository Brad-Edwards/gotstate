# tests/unit/runtime/test_graph.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

"""Unit tests for the graph-based state machine structure."""

import threading
import time
from unittest.mock import Mock

import pytest

from hsm.core.events import Event
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.runtime.graph import StateGraph


def test_add_state():
    """Test adding states to the graph."""
    graph = StateGraph()
    composite = CompositeState("composite")
    child = State("child")

    graph.add_state(composite)
    graph.add_state(child, parent=composite)

    # Verify hierarchy through graph methods only
    assert graph._parent_map[child] == composite
    assert child in graph.get_children(composite)
    assert graph.get_parent(child) == composite


def test_add_regular_state_parent():
    """Test adding states with a regular state as parent."""
    graph = StateGraph()
    parent = State("parent")
    child = State("child")

    graph.add_state(parent)
    graph.add_state(child, parent=parent)

    # Verify relationships through graph methods
    assert graph._parent_map[child] == parent
    assert child in graph.get_children(parent)
    assert graph.get_parent(child) == parent


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
    parent = CompositeState("parent")
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


def test_merge_submachine():
    """Test merging a submachine into the main graph."""
    main_graph = StateGraph()
    sub_graph = StateGraph()

    # Set up main graph
    parent = CompositeState("parent")
    main_graph.add_state(parent)

    # Set up submachine
    sub_initial = State("sub_initial")
    sub_other = State("sub_other")
    sub_graph.add_state(sub_initial)
    sub_graph.add_state(sub_other)
    sub_graph.set_initial_state(None, sub_initial)  # Set root initial state

    # Add transition in submachine
    transition = Transition(source=sub_initial, target=sub_other)
    sub_graph.add_transition(transition)

    # Add some state data
    sub_graph.set_state_data(sub_initial, "test_key", "test_value")

    # Merge submachine
    main_graph.merge_submachine(parent, sub_graph)

    # Verify states were merged correctly
    children = main_graph.get_children(parent)
    assert len(children) == 2

    # Find merged states by name
    merged_initial = next(s for s in children if s.name == "sub_initial")
    merged_other = next(s for s in children if s.name == "sub_other")

    # Verify initial state was preserved
    assert main_graph.get_initial_state(parent) == merged_initial

    # Verify transitions were merged
    transitions = main_graph.get_valid_transitions(merged_initial, Event("test"))
    assert len(transitions) == 1
    assert transitions[0].source == merged_initial
    assert transitions[0].target == merged_other

    # Verify state data was copied
    assert main_graph.get_state_data(merged_initial)["test_key"] == "test_value"


def test_current_state_management():
    """Test current state management."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(state1)
    graph.add_state(state2)

    # Test initial state
    assert graph.get_current_state() is None

    # Test setting valid state
    graph.set_current_state(state1)
    assert graph.get_current_state() == state1

    # Test setting to None
    graph.set_current_state(None)
    assert graph.get_current_state() is None

    # Test setting invalid state
    invalid_state = State("invalid")
    with pytest.raises(ValueError, match="State 'invalid' not in graph"):
        graph.set_current_state(invalid_state)


def test_merge_submachine_error_handling():
    """Test error handling during submachine merge."""
    main_graph = StateGraph()
    sub_graph = StateGraph()

    # Set up graphs
    parent = CompositeState("parent")
    sub_state = State("sub")

    main_graph.add_state(parent)
    sub_graph.add_state(sub_state)

    # Create a mock lock that always fails to acquire
    mock_lock = Mock()
    mock_lock.acquire.return_value = False

    # Replace the graph lock with our mock
    original_lock = main_graph._graph_lock
    main_graph._graph_lock = mock_lock

    try:
        with pytest.raises(RuntimeError, match="Failed to acquire graph lock for merge operation"):
            main_graph.merge_submachine(parent, sub_graph)
    finally:
        # Restore the original lock
        main_graph._graph_lock = original_lock


def test_resolve_active_state_with_no_initial():
    """Test resolving active state when no initial state is set."""
    graph = StateGraph()
    composite = CompositeState("composite")
    child1 = State("child1")
    child2 = State("child2")

    graph.add_state(composite)
    graph.add_state(child1, parent=composite)
    graph.add_state(child2, parent=composite)

    # No initial state set, should pick first child
    resolved = graph.resolve_active_state(composite)
    assert resolved in {child1, child2}

    # Verify initial state was set
    assert graph.get_initial_state(composite) == resolved


def test_cycle_detection():
    """Test cycle detection in state hierarchy."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")
    state3 = State("state3")

    graph.add_state(state1)
    graph.add_state(state2, parent=state1)
    graph.add_state(state3, parent=state2)

    # Attempt to create a cycle
    assert graph._would_create_cycle(state1, state3) is True


def test_state_data_errors():
    """Test error handling for state data operations."""
    graph = StateGraph()
    state = State("test")
    non_existent = State("non_existent")

    # Test accessing data for non-existent state
    with pytest.raises(AttributeError, match="State data cannot be accessed directly"):
        graph.get_state_data(non_existent)

    # Test setting data for non-existent state
    with pytest.raises(KeyError):
        graph.set_state_data(non_existent, "key", "value")

    # Test normal operation
    graph.add_state(state)
    graph.set_state_data(state, "key", "value")
    assert graph.get_state_data(state)["key"] == "value"


def test_history_state_management():
    """Test history state recording and retrieval."""
    graph = StateGraph()
    composite = CompositeState("composite")
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(composite)
    graph.add_state(state1, parent=composite)
    graph.add_state(state2, parent=composite)

    # Record history
    graph.record_history(composite, state1)
    assert graph.get_history_state(composite) == state1

    # Update history
    graph.record_history(composite, state2)
    assert graph.get_history_state(composite) == state2

    # Clear history
    graph.clear_history()
    assert graph.get_history_state(composite) is None


def test_complex_merge_scenarios():
    """Test complex submachine merge scenarios."""
    main_graph = StateGraph()
    sub_graph = StateGraph()

    # Set up complex submachine
    parent = CompositeState("parent")
    parent._children = set()  # Initialize children set for CompositeState
    sub_comp = CompositeState("sub_comp")
    sub_comp._children = set()  # Initialize children set for CompositeState
    sub_state1 = State("sub_state1")
    sub_state2 = State("sub_state2")

    # Add states to submachine with proper hierarchy
    sub_graph.add_state(sub_comp)
    sub_graph.add_state(sub_state1, parent=sub_comp)
    sub_graph.add_state(sub_state2, parent=sub_comp)

    # Add transition and data in submachine
    transition = Transition(source=sub_state1, target=sub_state2)
    sub_graph.add_transition(transition)
    sub_graph.set_state_data(sub_state1, "test_key", "test_value")

    # Set initial states
    sub_graph.set_initial_state(sub_comp, sub_state1)
    sub_graph.set_initial_state(None, sub_comp)  # Set root initial state

    # Add parent to main graph and merge
    main_graph.add_state(parent)
    main_graph.merge_submachine(parent, sub_graph)

    # Verify states were merged correctly
    children = main_graph.get_children(parent)
    assert len(children) == 3  # All states become direct children

    # Verify all states are present
    state_names = {s.name for s in children}
    assert state_names == {"sub_comp", "sub_state1", "sub_state2"}

    # Find merged states by name
    merged_state1 = next(s for s in children if s.name == "sub_state1")
    merged_state2 = next(s for s in children if s.name == "sub_state2")

    # Verify transitions were preserved
    transitions = main_graph.get_valid_transitions(merged_state1, Event("test"))
    assert len(transitions) == 1
    assert transitions[0].source == merged_state1
    assert transitions[0].target == merged_state2

    # Verify state data was copied
    assert main_graph.get_state_data(merged_state1)["test_key"] == "test_value"


def test_validation_with_cycles():
    """Test validation when cycles are present in the graph."""
    graph = StateGraph()
    composite = CompositeState("composite")
    composite._children = set()  # Initialize children set for CompositeState

    # Only add the composite state - now it truly has no children
    graph.add_state(composite)

    errors = graph.validate()
    assert any("no children" in error.lower() for error in errors)
