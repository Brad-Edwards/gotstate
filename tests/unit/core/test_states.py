# tests/unit/test_states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest

from gotstate.core.states import CompositeState, State
from gotstate.runtime.graph import StateGraph


def test_state_creation():
    """Test basic state creation and properties."""
    state = State("TestState")
    assert state.name == "TestState"
    assert not state.entry_actions
    assert not state.exit_actions


def test_state_actions():
    """Test adding entry and exit actions to states."""
    state = State("TestState")

    def action1():
        pass

    def action2():
        pass

    state.entry_actions.append(action1)
    state.exit_actions.append(action2)

    assert action1 in state.entry_actions
    assert action2 in state.exit_actions


def test_composite_state_children():
    """Test composite state child management through graph."""
    graph = StateGraph()
    cs = CompositeState("Composite")
    child1 = State("Child1")
    child2 = State("Child2")

    # Add states through graph
    graph.add_state(cs)
    graph.add_state(child1, parent=cs)
    graph.add_state(child2, parent=cs)

    # Verify relationships through graph
    children = graph.get_children(cs)
    assert child1 in children
    assert child2 in children
    assert graph._parent_map[child1] == cs
    assert graph._parent_map[child2] == cs


def test_state_data_isolation():
    """Test that state data is properly isolated between siblings."""
    graph = StateGraph()
    parent = CompositeState("Parent")
    state1 = State("State1")
    state2 = State("State2")

    graph.add_state(parent)
    graph.add_state(state1, parent=parent)
    graph.add_state(state2, parent=parent)

    # Set and verify data through graph only
    graph.set_state_data(state1, "test", "value1")
    graph.set_state_data(state2, "test", "value2")

    assert graph.get_state_data(state1)["test"] == "value1"
    assert graph.get_state_data(state2)["test"] == "value2"


def test_initial_state_management():
    """Test initial state management through graph."""
    graph = StateGraph()
    cs = CompositeState("Composite")
    initial = State("Initial")
    other = State("Other")

    graph.add_state(cs)
    graph.add_state(initial, parent=cs)
    graph.add_state(other, parent=cs)

    # Set and verify initial state through graph
    graph.set_initial_state(cs, initial)
    assert graph.get_initial_state(cs) == initial


def test_state_equality():
    """Test state equality comparison."""
    state1 = State("test")
    state2 = State("test")
    state3 = State("other")

    assert state1 == state2
    assert state1 != state3
    assert state1 != "test"  # Compare with non-State object


def test_state_hash():
    """Test state hash for dictionary usage."""
    state1 = State("test")
    state2 = State("test")
    state3 = State("other")

    # States with same name should have same hash
    assert hash(state1) == hash(state2)
    assert hash(state1) != hash(state3)

    # Test in dictionary
    state_dict = {state1: "value"}
    assert state2 in state_dict  # Should work due to same hash
    assert state3 not in state_dict


def test_state_data_access_error():
    """Test error when accessing state data without graph."""
    state = State("test")

    with pytest.raises(AttributeError, match="State data cannot be accessed directly"):
        _ = state.data


def test_composite_state_initial_state_no_graph():
    """Test initial_state property when no graph is set."""
    composite = CompositeState("composite")
    assert composite.initial_state is None


def test_composite_state_initial_state_with_graph():
    """Test initial_state property with graph."""
    graph = StateGraph()
    composite = CompositeState("composite")
    initial = State("initial")

    graph.add_state(composite)
    graph.add_state(initial, parent=composite)
    graph.set_initial_state(composite, initial)

    # Set graph reference manually (normally done by graph.add_state)
    composite._graph = graph

    assert composite.initial_state == initial


def test_state_with_actions():
    """Test state with entry and exit actions."""
    action_called = False

    def test_action():
        nonlocal action_called
        action_called = True

    state = State("test", entry_actions=[test_action], exit_actions=[test_action])

    assert len(state.entry_actions) == 1
    assert len(state.exit_actions) == 1

    state.entry_actions[0]()
    assert action_called

    action_called = False
    state.exit_actions[0]()
    assert action_called
