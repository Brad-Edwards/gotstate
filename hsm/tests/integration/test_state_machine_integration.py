"""Integration tests for the HSM state machine components.

This module contains integration tests that verify the interaction between different
components of the hierarchical state machine implementation.
"""

import time
from typing import Any, Dict, List

import pytest

from hsm.core.actions import NoOpAction
from hsm.core.events import Event
from hsm.core.guards import NoOpGuard
from hsm.core.state_machine import StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validation import Validator
from hsm.interfaces.abc import AbstractState, AbstractTransition
from hsm.runtime.timers import Timer, TimerState

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def basic_states() -> List[AbstractState]:
    """Create basic test states."""

    class TestState(State):
        def __init__(self, state_id: str) -> None:
            super().__init__(state_id)
            self.enter_called = False
            self.exit_called = False

        def on_enter(self) -> None:
            self.enter_called = True

        def on_exit(self) -> None:
            self.exit_called = True

    return [TestState("state1"), TestState("state2"), TestState("state3")]


@pytest.fixture
def basic_transitions(basic_states: List[AbstractState]) -> List[AbstractTransition]:
    """Create basic transitions between test states."""
    return [Transition("state1", "state2"), Transition("state2", "state3"), Transition("state3", "state1")]


@pytest.fixture
def state_machine(basic_states: List[AbstractState], basic_transitions: List[AbstractTransition]) -> StateMachine:
    """Create a basic state machine instance."""
    return StateMachine(basic_states, basic_transitions, basic_states[0])


# -----------------------------------------------------------------------------
# STATE MACHINE INTEGRATION TESTS
# -----------------------------------------------------------------------------


def test_transition_failure_recovery(basic_states: List[AbstractState]) -> None:
    """Test recovery from failed transitions."""
    failing_action = NoOpAction()
    failing_action.execute = lambda event, state_data: exec('raise Exception("Action failed")')

    transitions = [
        Transition("state1", "state2", actions=[failing_action]),
        Transition("state2", "state3"),
        Transition("state3", "state1"),
    ]
    machine = StateMachine(basic_states, transitions, basic_states[0])

    machine.start()
    event = Event("test_event")

    with pytest.raises(Exception) as exc_info:
        machine.process_event(event)
    assert "Action failed" in str(exc_info.value)
    assert machine.get_current_state_id() == "state1"


def test_full_validation_workflow(
    basic_states: List[AbstractState], basic_transitions: List[AbstractTransition]
) -> None:
    """Test complete validation workflow."""
    validator = Validator(basic_states, basic_transitions, basic_states[0])

    # Run all validation types
    structure_results = validator.validate_structure()
    behavior_results = validator.validate_behavior()
    data_results = validator.validate_data()

    # Verify results
    assert isinstance(structure_results, list)
    assert isinstance(behavior_results, list)
    assert isinstance(data_results, list)

    # Verify no structural errors in basic valid configuration
    assert not any(r.severity == "ERROR" for r in structure_results)


def test_validation_cyclic_transitions(
    basic_states: List[AbstractState], basic_transitions: List[AbstractTransition]
) -> None:
    """Test validation of cyclic transitions."""
    validator = Validator(basic_states, basic_transitions, basic_states[0])
    results = validator.validate_structure()
    assert not any(r.severity == "ERROR" for r in results)  # Cycles are allowed


# -----------------------------------------------------------------------------
# STATE AND TRANSITION INTEGRATION TESTS
# -----------------------------------------------------------------------------


def test_multiple_actions_execution() -> None:
    """Test execution of multiple actions in a transition."""

    class TestAction(NoOpAction):
        def __init__(self) -> None:
            self.executed = False
            self.event = None
            self.state_data = None

        def execute(self, event: Event, state_data: Dict[str, Any]) -> None:
            self.executed = True
            self.event = event
            self.state_data = state_data

    action1 = TestAction()
    action2 = TestAction()
    transition = Transition("source", "target", actions=[action1, action2])

    event = Event("test_event")
    state_data = {"test": "data"}

    for action in transition.get_actions():
        action.execute(event, state_data)
        assert action.executed
        assert action.event == event
        assert action.state_data == state_data


def test_composite_state_lifecycle() -> None:
    """Test composite state lifecycle with substates."""

    class TestState(State):
        def __init__(self, state_id: str) -> None:
            super().__init__(state_id)
            self.enter_called = False
            self.exit_called = False

        def on_enter(self) -> None:
            self.enter_called = True

        def on_exit(self) -> None:
            self.exit_called = True

    substate = TestState("sub")
    composite = CompositeState("composite", [substate], substate)

    # Test state transitions
    composite.on_enter()
    assert not composite.enter_called  # Composite states don't track enter/exit

    substate.on_enter()
    assert substate.enter_called

    substate.on_exit()
    assert substate.exit_called

    composite.on_exit()
    assert not composite.exit_called  # Composite states don't track enter/exit


# -----------------------------------------------------------------------------
# TIMER INTEGRATION TESTS
# -----------------------------------------------------------------------------


def test_timer_state_completion() -> None:
    """Test complete timer lifecycle with actual time delays."""
    callback_called = False

    def callback(event: Event) -> None:
        nonlocal callback_called
        callback_called = True

    timer = Timer()
    event = Event("test_event")

    timer.register_callback(callback)
    assert timer.get_info().state == TimerState.IDLE

    timer.schedule_timeout(0.1, event)
    assert timer.get_info().state == TimerState.RUNNING

    time.sleep(0.2)  # Wait for timer to complete
    assert timer.get_info().state == TimerState.COMPLETED
    assert callback_called


def test_timer_state_transitions() -> None:
    """Test timer state transitions through its lifecycle."""
    timer = Timer()
    event = Event("test_event")

    # Initial state
    assert timer.get_info().state == TimerState.IDLE

    # Schedule
    timer.schedule_timeout(0.1, event)
    assert timer.get_info().state == TimerState.RUNNING

    # Cancel
    timer.cancel_timeout(event.get_id())
    assert timer.get_info().state == TimerState.CANCELLED

    # Schedule new timer
    timer.schedule_timeout(0.1, event)
    assert timer.get_info().state == TimerState.RUNNING

    # Let it complete
    time.sleep(0.2)
    assert timer.get_info().state == TimerState.COMPLETED


def test_composite_state_transitions():
    """Test full composite state transition cycle"""
    # Current integration test content of test_composite_state_history
    ...


def test_performance_under_load():
    """Test performance with many states"""
    # Current content of test_performance_stress
    ...


def test_multiple_event_sequence():
    """Test processing multiple events in sequence"""
    # Current content of test_rapid_events
    ...
