from unittest.mock import Mock

import pytest

from gotstate.core.base import StateBase


def test_state_creation_minimal():
    """Test creating a state with minimal arguments."""
    state = StateBase(name="test")
    assert state.name == "test"
    assert state.parent is None
    assert state.entry_actions == []
    assert state.exit_actions == []


def test_state_creation_with_parent():
    """Test creating a state with a parent state."""
    parent = StateBase(name="parent")
    child = StateBase(name="child", parent=parent)
    assert child.parent == parent
    assert child.name == "child"


def test_state_creation_with_actions():
    """Test creating a state with entry and exit actions."""
    entry_action = Mock()
    exit_action = Mock()

    state = StateBase(name="test", entry_actions=[entry_action], exit_actions=[exit_action])

    assert state.entry_actions == [entry_action]
    assert state.exit_actions == [exit_action]


def test_on_enter_executes_actions():
    """Test that on_enter executes all entry actions in order."""
    action1 = Mock()
    action2 = Mock()

    state = StateBase(name="test", entry_actions=[action1, action2])

    state.on_enter()

    action1.assert_called_once()
    action2.assert_called_once()
    assert action1.call_count == 1
    assert action2.call_count == 1


def test_on_exit_executes_actions():
    """Test that on_exit executes all exit actions in order."""
    action1 = Mock()
    action2 = Mock()

    state = StateBase(name="test", exit_actions=[action1, action2])

    state.on_exit()

    action1.assert_called_once()
    action2.assert_called_once()
    assert action1.call_count == 1
    assert action2.call_count == 1


def test_state_equality():
    """Test state equality comparison."""
    state1 = StateBase(name="test")
    state2 = StateBase(name="test")  # Same name but different instance
    state3 = state1  # Same instance

    assert state1 != state2  # Different instances should not be equal
    assert state1 == state3  # Same instance should be equal
    assert state1 != "not_a_state"  # Different types should not be equal


def test_state_hash():
    """Test state hash behavior."""
    state1 = StateBase(name="test")
    state2 = StateBase(name="test")  # Same name but different instance

    # Different instances with same name should have different hashes
    assert hash(state1) != hash(state2)

    # Same instance should maintain consistent hash
    assert hash(state1) == hash(state1)

    # States can be used as dictionary keys
    state_dict = {state1: "value1", state2: "value2"}
    assert len(state_dict) == 2
    assert state_dict[state1] == "value1"
    assert state_dict[state2] == "value2"


def test_hierarchical_state_structure():
    """Test creating and validating a hierarchical state structure."""
    root = StateBase(name="root")
    child1 = StateBase(name="child1", parent=root)
    child2 = StateBase(name="child2", parent=root)
    grandchild = StateBase(name="grandchild", parent=child1)

    assert child1.parent == root
    assert child2.parent == root
    assert grandchild.parent == child1
    assert root.parent is None


def test_action_execution_order():
    """Test that actions are executed in the order they were added."""
    call_order = []

    def action1():
        call_order.append(1)

    def action2():
        call_order.append(2)

    def action3():
        call_order.append(3)

    state = StateBase(name="test", entry_actions=[action1, action2, action3], exit_actions=[action3, action2, action1])

    state.on_enter()
    assert call_order == [1, 2, 3]

    call_order.clear()

    state.on_exit()
    assert call_order == [3, 2, 1]


def test_empty_actions():
    """Test that states with no actions handle enter/exit gracefully."""
    state = StateBase(name="test")

    # Should not raise any exceptions
    state.on_enter()
    state.on_exit()


def test_action_independence():
    """Test that actions between states don't interfere with each other."""
    action1 = Mock()
    action2 = Mock()

    state1 = StateBase(name="state1", entry_actions=[action1])
    state2 = StateBase(name="state2", entry_actions=[action2])

    state1.on_enter()
    assert action1.call_count == 1
    assert action2.call_count == 0

    state2.on_enter()
    assert action1.call_count == 1
    assert action2.call_count == 1
