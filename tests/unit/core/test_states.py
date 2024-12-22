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
    cs._children = set()
    child1 = State("Child1")
    child2 = State("Child2")

    # Add states through graph
    graph.add_state(cs)
    graph.add_state(child1, parent=cs)
    graph.add_state(child2, parent=cs)

    # Verify relationships
    assert child1 in cs._children
    assert child2 in cs._children
    assert child1.parent == cs
    assert child2.parent == cs


def test_state_data_isolation():
    """Test that state data is properly isolated between siblings."""
    graph = StateGraph()
    parent = CompositeState("Parent")
    parent._children = set()
    state1 = State("State1")
    state2 = State("State2")

    # Set data before adding to graph
    state1.data["test"] = "value1"
    state2.data["test"] = "value2"

    # Add states through graph
    graph.add_state(parent)
    graph.add_state(state1, parent=parent)
    graph.add_state(state2, parent=parent)

    # Verify data isolation
    assert state1.data["test"] == "value1"
    assert state2.data["test"] == "value2"
    assert state1.data is not state2.data  # Ensure different dict instances
