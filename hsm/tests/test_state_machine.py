# hsm/tests/test_state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from hsm.core.actions import NoOpAction
from hsm.core.errors import ConfigurationError, HSMError, InvalidStateError, InvalidTransitionError
from hsm.core.events import Event
from hsm.core.guards import NoOpGuard
from hsm.core.hooks import HookManager
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.transitions import Transition
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractTransition

# -----------------------------------------------------------------------------
# TEST FIXTURES AND HELPERS

# -----------------------------------------------------------------------------


@pytest.fixture
def state_fixture() -> type:
    """Fixture that returns a TestState class."""

    class _TestState(State):
        def __init__(self, state_id: str) -> None:
            super().__init__(state_id)
            self.enter_called = False
            self.exit_called = False

        def on_enter(self) -> None:
            self.enter_called = True

        def on_exit(self) -> None:
            self.exit_called = True

    return _TestState


@pytest.fixture
def basic_states(state_fixture) -> List[AbstractState]:
    """Create a basic set of test states."""
    return [state_fixture("state1"), state_fixture("state2"), state_fixture("state3")]


@pytest.fixture
def basic_transitions(basic_states: List[AbstractState]) -> List[AbstractTransition]:
    """Create basic transitions between test states."""
    return [Transition("state1", "state2"), Transition("state2", "state3"), Transition("state3", "state1")]


@pytest.fixture
def state_machine(basic_states: List[AbstractState], basic_transitions: List[AbstractTransition]) -> StateMachine:
    """Create a basic state machine instance."""
    return StateMachine(basic_states, basic_transitions, basic_states[0])


# -----------------------------------------------------------------------------
# INITIALIZATION TESTS
# -----------------------------------------------------------------------------


def test_state_machine_init_empty_states() -> None:
    """Test that initializing with empty states raises ValueError."""
    with pytest.raises(ValueError, match="States list cannot be empty"):
        StateMachine([], [], None)


def test_state_machine_init_empty_transitions(state_fixture) -> None:
    """Test that initializing with empty transitions raises ValueError."""
    states = [state_fixture("state1")]
    with pytest.raises(ValueError, match="Transitions list cannot be empty"):
        StateMachine(states, [], states[0])


def test_state_machine_init_invalid_initial_state(state_fixture) -> None:
    """Test that initializing with invalid initial state raises ValueError."""
    states = [state_fixture("state1")]
    transitions = [Transition("state1", "state1")]
    invalid_initial = state_fixture("state2")

    with pytest.raises(ValueError, match="Initial state must be in states list"):
        StateMachine(states, transitions, invalid_initial)


def test_state_machine_init_success(state_machine: StateMachine) -> None:
    """Test successful state machine initialization."""
    assert not state_machine._running
    assert state_machine._current_state is None
    assert isinstance(state_machine._hook_manager, HookManager)


# -----------------------------------------------------------------------------
# START/STOP TESTS
# -----------------------------------------------------------------------------


def test_start_already_running(state_machine: StateMachine) -> None:
    """Test that starting an already running machine raises InvalidStateError."""
    state_machine.start()

    with pytest.raises(InvalidStateError) as exc_info:
        state_machine.start()
    assert exc_info.value.operation == "start"


def test_stop_not_running(state_machine: StateMachine) -> None:
    """Test that stopping a non-running machine raises InvalidStateError."""
    with pytest.raises(InvalidStateError) as exc_info:
        state_machine.stop()
    assert exc_info.value.operation == "stop"


def test_start_stop_cycle(state_machine: StateMachine) -> None:
    """Test normal start/stop cycle behavior."""
    state_machine.start()
    assert state_machine._running
    assert state_machine._current_state is not None
    assert state_machine._current_state.enter_called

    state_machine.stop()
    assert not state_machine._running
    assert state_machine._current_state is None


# -----------------------------------------------------------------------------
# EVENT PROCESSING TESTS
# -----------------------------------------------------------------------------


def test_process_event_machine_stopped(state_machine: StateMachine) -> None:
    """Test that processing events while stopped raises InvalidStateError."""
    event = Event("test_event")
    with pytest.raises(InvalidStateError) as exc_info:
        state_machine.process_event(event)
    assert exc_info.value.operation == "process_event"


def test_process_event_no_matching_transition(state_machine: StateMachine) -> None:
    """Test handling of events with no matching transitions."""
    state_machine.start()
    event = Event("no_matching_transition")
    # Should not raise, just return without transition
    state_machine.process_event(event)


def test_process_event_with_guard(basic_states: List[AbstractState]) -> None:
    """Test event processing with guard conditions."""
    guard = NoOpGuard()
    transitions = [
        Transition("state1", "state2", guard=guard),
        Transition("state2", "state1"),
        Transition("state2", "state3"),  # Add connection to state3
        Transition("state3", "state1"),  # Add return path
    ]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    event = Event("test_event")
    machine.process_event(event)
    assert machine.get_current_state_id() == "state2"


def test_process_event_with_actions(basic_states: List[AbstractState]) -> None:
    """Test event processing with transition actions."""
    action = MagicMock(spec=NoOpAction)
    transitions = [
        Transition("state1", "state2", actions=[action]),
        Transition("state2", "state1"),
        Transition("state2", "state3"),  # Add connection to state3
        Transition("state3", "state1"),  # Add return path
    ]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    event = Event("test_event")
    machine.process_event(event)
    action.execute.assert_called_once()


# -----------------------------------------------------------------------------
# TRANSITION HANDLING TESTS
# -----------------------------------------------------------------------------


def test_transition_priority(basic_states: List[AbstractState]) -> None:
    """Test that highest priority transition is selected."""
    transitions = [Transition("state1", "state2", priority=1), Transition("state1", "state3", priority=2)]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    event = Event("test_event")
    machine.process_event(event)
    assert machine.get_current_state_id() == "state3"


def test_transition_failure_recovery(basic_states: List[AbstractState]) -> None:
    """Test recovery from failed transitions."""
    failing_action = MagicMock(spec=NoOpAction)
    failing_action.execute.side_effect = Exception("Action failed")

    transitions = [
        Transition("state1", "state2", actions=[failing_action]),
        Transition("state2", "state3"),  # Add connection to state3
        Transition("state3", "state1"),  # Add return path
    ]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    event = Event("test_event")

    with pytest.raises(InvalidTransitionError):
        machine.process_event(event)
    assert machine.get_current_state_id() == "state1"


# -----------------------------------------------------------------------------
# STATE DATA MANAGEMENT TESTS
# -----------------------------------------------------------------------------


def test_state_data_isolation(basic_states: List[AbstractState]) -> None:
    """Test that state data remains isolated during transitions."""
    transitions = [
        Transition("state1", "state2"),
        Transition("state2", "state1"),
        Transition("state2", "state3"),  # Add connection to state3
        Transition("state3", "state1"),  # Add return path
    ]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    state1 = machine._states["state1"]
    state1.data["test"] = "value"

    event = Event("test_event")
    machine.process_event(event)

    state2 = machine._states["state2"]
    assert "test" not in state2.data


# -----------------------------------------------------------------------------
# VALIDATION TESTS
# -----------------------------------------------------------------------------


def test_validation_unreachable_state(basic_states: List[AbstractState]) -> None:
    """Test validation catches unreachable states."""
    transitions = [Transition("state1", "state2")]  # state3 unreachable

    with pytest.raises(ConfigurationError) as exc_info:
        StateMachine(basic_states, transitions, basic_states[0])
    assert "structure" in exc_info.value.component


def test_validation_invalid_transition_target(basic_states: List[AbstractState]) -> None:
    """Test validation catches invalid transition targets."""
    transitions = [Transition("state1", "nonexistent")]

    with pytest.raises(ConfigurationError) as exc_info:
        StateMachine(basic_states, transitions, basic_states[0])
    assert "structure" in exc_info.value.component


# -----------------------------------------------------------------------------
# HOOK TESTS
# -----------------------------------------------------------------------------


def test_hooks_called(state_machine: StateMachine) -> None:
    """Test that hooks are called during transitions."""
    mock_hook = MagicMock()
    state_machine._hook_manager.register_hook(mock_hook)

    state_machine.start()
    event = Event("test_event")
    state_machine.process_event(event)

    assert mock_hook.pre_transition.called
    assert mock_hook.post_transition.called


# -----------------------------------------------------------------------------
# ERROR HANDLING TESTS
# -----------------------------------------------------------------------------


def test_error_context_preserved(basic_states: List[AbstractState]) -> None:
    """Test that error context is preserved in exceptions."""
    failing_action = MagicMock(spec=NoOpAction)
    failing_action.execute.side_effect = Exception("Custom error")

    # Include all necessary transitions to make the state machine valid
    transitions = [
        Transition("state1", "state2", actions=[failing_action]),
        Transition("state2", "state3"),
        Transition("state3", "state1"),
    ]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    event = Event("test_event")

    with pytest.raises(InvalidTransitionError) as exc_info:
        machine.process_event(event)

    # Verify error context is preserved
    assert "Custom error" in str(exc_info.value)
    assert exc_info.value.source_state == "state1"
    assert exc_info.value.target_state == "state2"
    assert isinstance(exc_info.value.details, dict)
    assert "error" in exc_info.value.details
    assert "Custom error" in exc_info.value.details["error"]


# -----------------------------------------------------------------------------
# CLEANUP TESTS
# -----------------------------------------------------------------------------


def test_cleanup_on_stop(state_machine: StateMachine) -> None:
    """Test proper cleanup when stopping the state machine."""
    state_machine.start()
    initial_state = state_machine._current_state
    assert initial_state is not None

    state_machine.stop()
    assert state_machine._current_state is None
    assert not state_machine._running
    assert initial_state.exit_called


def test_reset_cleanup(state_machine: StateMachine) -> None:
    """Test cleanup during reset operation."""
    state_machine.start()
    state = state_machine._current_state
    assert state is not None

    state_machine.reset()
    assert state.exit_called
    assert state_machine._current_state is not None
    assert state_machine._current_state.enter_called
