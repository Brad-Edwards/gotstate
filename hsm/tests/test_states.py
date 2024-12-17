# hsm/tests/test_states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from copy import deepcopy
from typing import Any, Dict, List

import pytest

from hsm.core.states import CompositeState, State
from hsm.interfaces.abc import AbstractCompositeState, AbstractState
from hsm.interfaces.types import StateID

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class ConcreteState(State):
    """A concrete implementation of State for testing."""

    def __init__(self, state_id: str):
        super().__init__(state_id)
        self.enter_called = False
        self.exit_called = False

    def on_enter(self) -> None:
        self.enter_called = True

    def on_exit(self) -> None:
        self.exit_called = True


class ConcreteCompositeState(CompositeState):
    """A concrete implementation of CompositeState for testing."""

    def __init__(
        self, state_id: str, substates: List[AbstractState], initial_state: AbstractState, has_history: bool = False
    ):
        super().__init__(state_id, substates, initial_state, has_history)
        self.enter_called = False
        self.exit_called = False


# -----------------------------------------------------------------------------
# STATE BASE CLASS TESTS
# -----------------------------------------------------------------------------


def test_state_initialization():
    """Test state initialization with valid and invalid IDs."""
    state = ConcreteState("test_state")
    assert state.get_id() == "test_state"
    assert isinstance(state.data, dict)
    assert len(state.data) == 0

    with pytest.raises(ValueError, match="state_id cannot be empty"):
        ConcreteState("")


def test_state_abstract_methods():
    """Test that State class enforces implementation of abstract methods."""

    class IncompleteState(State):
        pass

    state = IncompleteState("test")
    with pytest.raises(NotImplementedError, match="on_enter must be implemented"):
        state.on_enter()

    with pytest.raises(NotImplementedError, match="on_exit must be implemented"):
        state.on_exit()


def test_state_data_manipulation():
    """Test state data dictionary manipulation."""
    state = ConcreteState("test")

    # Test modifying dictionary contents
    state.data["key"] = "value"
    assert state.data["key"] == "value"

    # Test that the dictionary reference itself is maintained
    original_data_id = id(state.data)
    state.data.clear()
    assert id(state.data) == original_data_id


# -----------------------------------------------------------------------------
# COMPOSITE STATE TESTS
# -----------------------------------------------------------------------------


def test_composite_state_initialization():
    """Test composite state initialization with various configurations."""
    substate = ConcreteState("sub")

    # Valid initialization
    state = ConcreteCompositeState("test", [substate], substate, True)
    assert state.get_id() == "test"
    assert state.has_history() is True
    assert state.get_substates() == [substate]
    assert state.get_initial_state() == substate

    # Empty state_id
    with pytest.raises(ValueError, match="state_id cannot be empty"):
        ConcreteCompositeState("", [substate], substate)

    # Invalid substates type
    with pytest.raises(ValueError, match="substates must be a list"):
        ConcreteCompositeState("test", None, substate)  # type: ignore

    # Initial state not in substates
    other_state = ConcreteState("other")
    with pytest.raises(ValueError, match="initial_state must be one of the substates"):
        ConcreteCompositeState("test", [substate], other_state)


def test_composite_state_history():
    """Test composite state history tracking."""
    initial_state = ConcreteState("initial")
    other_state = ConcreteState("other")
    state = ConcreteCompositeState("test", [initial_state, other_state], initial_state, has_history=True)

    # Initially returns initial state
    assert state.get_initial_state() == initial_state

    # After setting last active, returns that instead
    state.set_last_active(other_state)
    assert state.get_initial_state() == other_state

    # Invalid state for last active
    invalid_state = ConcreteState("invalid")
    with pytest.raises(ValueError, match="state must be a substate"):
        state.set_last_active(invalid_state)


def test_composite_state_without_history():
    """Test composite state behavior when history is disabled."""
    initial_state = ConcreteState("initial")
    other_state = ConcreteState("other")
    state = ConcreteCompositeState("test", [initial_state, other_state], initial_state, has_history=False)

    # Always returns initial state
    assert state.get_initial_state() == initial_state
    state.set_last_active(other_state)
    assert state.get_initial_state() == initial_state


def test_nested_composite_states():
    """Test nested composite state structures."""
    inner_state = ConcreteState("inner")
    middle_state = ConcreteCompositeState("middle", [inner_state], inner_state)
    outer_state: ConcreteCompositeState = ConcreteCompositeState("outer", [middle_state], middle_state)

    assert len(outer_state.get_substates()) == 1
    assert isinstance(outer_state.get_substates()[0], CompositeState)
    assert outer_state.get_substates()[0].get_substates()[0].get_id() == "inner"

    # Test history propagation
    outer_state.set_last_active(middle_state)
    middle_state.set_last_active(inner_state)
    assert outer_state.get_initial_state() == middle_state
    assert middle_state.get_initial_state() == inner_state


# -----------------------------------------------------------------------------
# EDGE CASES AND ERROR CONDITIONS
# -----------------------------------------------------------------------------


def test_empty_composite_state():
    """Test composite state with no substates."""
    # When substates is empty, initial_state must be None
    state = ConcreteCompositeState("empty", [], None)
    assert len(state.get_substates()) == 0
    assert state.get_initial_state() is None


def test_state_lifecycle():
    """Test state lifecycle methods."""
    state = ConcreteState("test")
    assert not state.enter_called
    assert not state.exit_called

    state.on_enter()
    assert state.enter_called
    assert not state.exit_called

    state.on_exit()
    assert state.enter_called
    assert state.exit_called


def test_composite_state_lifecycle():
    """Test composite state lifecycle methods."""
    substate = ConcreteState("sub")
    state = ConcreteCompositeState("test", [substate], substate)

    # Composite state's on_enter/on_exit are no-ops
    state.on_enter()  # Should not raise
    state.on_exit()  # Should not raise


# -----------------------------------------------------------------------------
# PROTOCOL COMPLIANCE TESTS
# -----------------------------------------------------------------------------


def test_state_protocol_compliance():
    """Test that State implementations comply with AbstractState protocol."""
    state = ConcreteState("test")
    assert isinstance(state, AbstractState)

    # Verify all required methods are present and callable
    state.on_enter()
    state.on_exit()
    _ = state.data
    _ = state.get_id()


def test_composite_state_protocol_compliance():
    """Test that CompositeState implementations comply with AbstractCompositeState protocol."""
    substate = ConcreteState("sub")
    state = ConcreteCompositeState("test", [substate], substate)

    assert isinstance(state, AbstractCompositeState)
    assert isinstance(state, State)

    # Verify all required methods are present and callable
    state.on_enter()
    state.on_exit()
    _ = state.data
    _ = state.get_id()
    _ = state.get_substates()
    _ = state.get_initial_state()
    _ = state.has_history()


# -----------------------------------------------------------------------------
# DATA MANAGEMENT TESTS
# -----------------------------------------------------------------------------


def test_state_data_isolation():
    """Test that state data is properly isolated between states."""
    state1 = ConcreteState("state1")
    state2 = ConcreteState("state2")

    # Verify data dictionaries are separate
    state1.data["key"] = "value1"
    state2.data["key"] = "value2"
    assert state1.data["key"] == "value1"
    assert state2.data["key"] == "value2"
    assert state1.data is not state2.data

    # Verify nested data structures are isolated
    state1.data["nested"] = {"a": 1}
    state2.data["nested"] = {"a": 2}
    state1.data["nested"]["a"] = 3
    assert state2.data["nested"]["a"] == 2


def test_composite_state_data_isolation():
    """Test that composite states maintain data isolation from substates."""
    substate = ConcreteState("sub")
    composite = ConcreteCompositeState("composite", [substate], substate)

    # Verify composite state data is isolated from substate
    composite["shared_key"] = "parent"  # Use __setitem__
    substate["shared_key"] = "child"  # Use __setitem__

    # Use __getitem__ to access values
    assert composite["shared_key"] == "parent"
    assert substate["shared_key"] == "child"
    assert composite.data is not substate.data


def test_state_data_immutability():
    """Test that state data references cannot be reassigned."""
    state = ConcreteState("test")
    original_data = state.data

    # Try to assign new dictionary
    with pytest.raises(AttributeError):
        state.data = {}  # type: ignore

    # Verify original data is unchanged
    assert state.data is original_data


def test_state_data_deep_copy():
    """Test that nested data structures maintain isolation."""
    state1 = ConcreteState("state1")
    state2 = ConcreteState("state2")

    # Set up nested data structure
    nested_data = {"list": [1, 2, 3], "dict": {"a": 1}}
    state1["nested"] = nested_data  # Use __setitem__
    state2["nested"] = nested_data  # Use __setitem__

    # Get a copy of the nested data
    state1_data = state1["nested"]  # Use __getitem__
    state1_data["list"].append(4)
    state1_data["dict"]["b"] = 2

    # Verify state2's data is unchanged
    state2_data = state2["nested"]  # Use __getitem__
    assert 4 not in state2_data["list"]
    assert "b" not in state2_data["dict"]


# -----------------------------------------------------------------------------
# ERROR HANDLING AND VALIDATION TESTS
# -----------------------------------------------------------------------------


def test_state_id_validation():
    """Test state ID validation rules."""
    # Empty string
    with pytest.raises(ValueError, match="state_id cannot be empty"):
        ConcreteState("")

    # None
    with pytest.raises(TypeError):
        ConcreteState(None)  # type: ignore

    # Whitespace only
    with pytest.raises(ValueError, match="state_id cannot be empty"):
        ConcreteState("   ")


def test_composite_state_validation():
    """Test composite state validation rules."""
    state1 = ConcreteState("state1")
    state2 = ConcreteState("state2")

    # None substates
    with pytest.raises(ValueError, match="substates must be a list"):
        ConcreteCompositeState("test", None, state1)  # type: ignore

    # Non-list substates
    with pytest.raises(ValueError, match="substates must be a list"):
        ConcreteCompositeState("test", "not_a_list", state1)  # type: ignore

    # Initial state not in substates
    with pytest.raises(ValueError, match="initial_state must be one of the substates"):
        ConcreteCompositeState("test", [state1], state2)

    # Empty state_id
    with pytest.raises(ValueError, match="state_id cannot be empty"):
        ConcreteCompositeState("", [state1], state1)


def test_composite_state_history_validation():
    """Test history-related validation in composite states."""
    state = ConcreteState("test")
    composite = ConcreteCompositeState("composite", [state], state, has_history=True)

    # Set invalid state as last active
    invalid_state = ConcreteState("invalid")
    with pytest.raises(ValueError, match="state must be a substate"):
        composite.set_last_active(invalid_state)

    # Set None as last active
    with pytest.raises(ValueError, match="state must be a substate"):
        composite.set_last_active(None)  # type: ignore


def test_composite_state_empty_validation():
    """Test validation of empty composite states."""
    state = ConcreteState("test")

    # Empty substates list should require None initial state
    with pytest.raises(ValueError):
        CompositeState("test", [], state)

    # Empty substates list with None initial state
    composite = CompositeState("test", [], None)
    assert len(composite.get_substates()) == 0
    assert composite.get_initial_state() is None


def test_composite_state_duplicate_substates():
    """Test handling of duplicate substates."""
    state = ConcreteState("test")

    # Same state instance twice in substates
    composite = ConcreteCompositeState("test", [state, state], state)
    assert len(composite.get_substates()) == 2
    assert composite.get_substates().count(state) == 2

    # Verify last_active still works with duplicates
    composite.set_last_active(state)
    assert composite.get_initial_state() is state


# -----------------------------------------------------------------------------
# HIERARCHICAL STATE TESTS
# -----------------------------------------------------------------------------


def test_deep_nested_states():
    """Test deeply nested state hierarchies."""
    # Create a deep hierarchy: outer -> middle -> inner -> leaf
    leaf_state = ConcreteState("leaf")
    inner_state = ConcreteCompositeState("inner", [leaf_state], leaf_state, True)
    middle_state = ConcreteCompositeState("middle", [inner_state], inner_state, True)
    outer_state = ConcreteCompositeState("outer", [middle_state], middle_state, True)

    # Test navigation through hierarchy
    assert outer_state.get_substates()[0] is middle_state
    assert middle_state.get_substates()[0] is inner_state
    assert inner_state.get_substates()[0] is leaf_state

    # Test history propagation through levels
    outer_state.set_last_active(middle_state)
    middle_state.set_last_active(inner_state)
    inner_state.set_last_active(leaf_state)

    assert outer_state.get_initial_state() is middle_state
    assert middle_state.get_initial_state() is inner_state
    assert inner_state.get_initial_state() is leaf_state


def test_sibling_states():
    """Test states with multiple siblings at each level."""
    # Create states at leaf level
    leaf1 = ConcreteState("leaf1")
    leaf2 = ConcreteState("leaf2")

    # Create inner level with multiple states
    inner1 = ConcreteCompositeState("inner1", [leaf1, leaf2], leaf1, True)
    inner2 = ConcreteCompositeState("inner2", [leaf1, leaf2], leaf2, True)

    # Create outer level containing all inner states
    outer = ConcreteCompositeState("outer", [inner1, inner2], inner1, True)

    # Test sibling relationships
    assert len(outer.get_substates()) == 2
    assert len(inner1.get_substates()) == 2
    assert len(inner2.get_substates()) == 2

    # Test history with siblings
    outer.set_last_active(inner2)
    inner1.set_last_active(leaf2)
    inner2.set_last_active(leaf1)

    assert outer.get_initial_state() is inner2
    assert inner1.get_initial_state() is leaf2
    assert inner2.get_initial_state() is leaf1


def test_mixed_history_states():
    """Test hierarchies with mixed history settings."""
    # Create states with different history settings
    leaf = ConcreteState("leaf")
    inner_no_history = ConcreteCompositeState("inner_no", [leaf], leaf, False)
    inner_with_history = ConcreteCompositeState("inner_yes", [leaf], leaf, True)
    outer = ConcreteCompositeState("outer", [inner_no_history, inner_with_history], inner_no_history, True)

    # Set last active states
    outer.set_last_active(inner_with_history)
    inner_no_history.set_last_active(leaf)
    inner_with_history.set_last_active(leaf)

    # Test history behavior
    assert outer.get_initial_state() is inner_with_history  # Has history
    assert inner_no_history.get_initial_state() is leaf  # Initial state
    assert inner_with_history.get_initial_state() is leaf  # Last active


def test_shared_substates():
    """Test handling of states shared between multiple parents."""
    # Create shared states
    shared_leaf = ConcreteState("shared")
    unique_leaf = ConcreteState("unique")

    # Create parents sharing a state
    parent1 = ConcreteCompositeState("parent1", [shared_leaf, unique_leaf], shared_leaf)
    parent2 = ConcreteCompositeState("parent2", [shared_leaf], shared_leaf)

    # Test that the same state can be in multiple parents
    assert shared_leaf in parent1.get_substates()
    assert shared_leaf in parent2.get_substates()

    # Test that history tracking works independently
    parent1.set_last_active(unique_leaf)
    parent2.set_last_active(shared_leaf)

    assert parent1.get_initial_state() is shared_leaf  # No history
    assert parent2.get_initial_state() is shared_leaf  # No history
