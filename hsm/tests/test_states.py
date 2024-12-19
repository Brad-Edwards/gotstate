# hsm/tests/test_states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import Any, Dict, List, Optional

import pytest

from hsm.core.states import CompositeState, State
from hsm.interfaces.abc import AbstractState
from hsm.interfaces.types import StateID

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class ConcreteState(State):
    """A concrete implementation of State for testing."""

    def __init__(self, state_id: str) -> None:
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
        self,
        state_id: str,
        substates: List[AbstractState],
        initial_state: Optional[AbstractState],
        has_history: bool = False,
    ) -> None:
        super().__init__(state_id, substates, initial_state, has_history)
        self.enter_called = False
        self.exit_called = False

    def on_enter(self) -> None:
        self.enter_called = True

    def on_exit(self) -> None:
        self.exit_called = True


# -----------------------------------------------------------------------------
# BASIC STATE TESTS
# -----------------------------------------------------------------------------


def test_state_initialization() -> None:
    """Test basic state initialization."""
    state = ConcreteState("test")
    assert state.get_id() == "test"
    assert isinstance(state.data, dict)
    assert len(state.data) == 0


def test_state_data_management() -> None:
    """Test state data dictionary management."""
    state = ConcreteState("test")
    state.data["key"] = "value"
    assert state.data["key"] == "value"


def test_state_data_isolation() -> None:
    """Test that each state has its own data dictionary."""
    state1 = ConcreteState("state1")
    state2 = ConcreteState("state2")

    state1.data["key"] = "value1"
    state2.data["key"] = "value2"

    assert state1.data["key"] == "value1"
    assert state2.data["key"] == "value2"


# -----------------------------------------------------------------------------
# COMPOSITE STATE TESTS
# -----------------------------------------------------------------------------


def test_composite_state_initialization() -> None:
    """Test composite state initialization."""
    substate = ConcreteState("sub")
    state = ConcreteCompositeState("test", [substate], substate)

    assert state.get_id() == "test"
    assert len(state.get_substates()) == 1
    assert state.get_initial_state() == substate


def test_composite_state_substate_access() -> None:
    """Test substate access methods."""
    substate1 = ConcreteState("sub1")
    substate2 = ConcreteState("sub2")
    state = ConcreteCompositeState("test", [substate1, substate2], substate1)

    assert len(state.get_substates()) == 2
    assert substate1 in state.get_substates()
    assert substate2 in state.get_substates()


def test_composite_state_history() -> None:
    """Test composite state history functionality."""
    substate = ConcreteState("sub")
    state = ConcreteCompositeState("test", [substate], substate, has_history=True)

    assert state.has_history()
    state.set_last_active(substate)
    assert state.get_last_active() == substate


# -----------------------------------------------------------------------------
# EDGE CASES AND ERROR CONDITIONS
# -----------------------------------------------------------------------------


def test_empty_composite_state() -> None:
    """Test composite state with no substates."""
    # When substates is empty, initial_state must be None
    state = ConcreteCompositeState("empty", [], None)
    assert len(state.get_substates()) == 0
    assert state.get_initial_state() is None


def test_composite_state_validation() -> None:
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


def test_composite_state_history_validation() -> None:
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


def test_composite_state_empty_validation() -> None:
    """Test validation of empty composite states."""
    state = ConcreteState("test")

    # Empty substates list should require None initial state
    with pytest.raises(ValueError):
        CompositeState("test", [], state)

    # Empty substates list with None initial state
    composite = CompositeState("test", [], None)
    assert len(composite.get_substates()) == 0
    assert composite.get_initial_state() is None


def test_composite_state_duplicate_substates() -> None:
    """Test handling of duplicate substates."""
    state = ConcreteState("test")

    # Same state instance twice in substates
    composite = ConcreteCompositeState("test", [state, state], state)
    assert len(composite.get_substates()) == 2
    assert composite.get_substates().count(state) == 2

    # Verify last_active still works with duplicates
    composite.set_last_active(state)
    assert composite.get_initial_state() is state


def test_state_id_validation() -> None:
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
