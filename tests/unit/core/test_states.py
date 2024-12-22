# tests/unit/test_states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.states import CompositeState, State
from hsm.runtime.graph import StateGraph


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

    # Add states to graph
    graph.add_state(parent)
    graph.add_state(state1, parent=parent)
    graph.add_state(state2, parent=parent)

    # Set data through graph
    graph.set_state_data(state1, "test", "value1")
    graph.set_state_data(state2, "test", "value2")

    # Verify data isolation
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
