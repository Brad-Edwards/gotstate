# tests/unit/test_state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import Mock, call, patch

import pytest

from hsm.core.events import Event
from hsm.core.state_machine import CompositeStateMachine, StateMachine, _ErrorRecoveryStrategy
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import ValidationError
from hsm.runtime.graph import StateGraph


class TestErrorRecoveryStrategy(_ErrorRecoveryStrategy):
    """Test implementation of error recovery strategy."""

    called = False
    last_error = None

    def recover(self, error: Exception, state_machine: StateMachine) -> None:
        TestErrorRecoveryStrategy.called = True
        TestErrorRecoveryStrategy.last_error = error


@pytest.fixture
def basic_machine():
    """Create a basic state machine setup."""
    state1 = State("state1")
    state2 = State("state2")

    # Create machine with initial state
    machine = StateMachine(state1)
    machine._set_current_state(state1)  # Set current state before validation

    # Add states to graph
    machine.add_state(state2)

    # Set up transition
    transition = Transition(source=state1, target=state2)
    machine.add_transition(transition)

    return machine, state1, state2, transition


def test_initial_state(basic_machine):
    """Test that machine starts in the initial state."""
    machine, state1, _, _ = basic_machine
    machine.start()
    assert machine.current_state == state1


def test_state_transition(basic_machine):
    """Test basic state transition."""
    machine, state1, state2, _ = basic_machine

    # Mock state methods to verify calls
    state1.on_exit = Mock()
    state2.on_enter = Mock()

    machine.start()
    event = Event("test")
    assert machine.process_event(event)  # Should return True

    state1.on_exit.assert_called_once()
    state2.on_enter.assert_called_once()
    assert machine.current_state == state2


def test_no_valid_transition():
    """Test that invalid events are handled properly."""
    state = State("state")
    machine = StateMachine(state)
    machine._set_current_state(state)  # Set current state before validation
    machine.start()

    event = Event("unknown")
    assert not machine.process_event(event)  # Should return False
    assert machine.current_state == state


def test_error_recovery():
    """Test error recovery strategy."""
    state1 = State("state1")
    state2 = State("state2")

    recovery = TestErrorRecoveryStrategy()
    machine = StateMachine(state1, error_recovery=recovery)
    machine._set_current_state(state1)  # Set current state before validation
    machine.add_state(state2)

    # Add transition that will raise an exception
    def failing_action(event):
        raise ValueError("Test error")

    transition = Transition(source=state1, target=state2, actions=[failing_action])
    machine.add_transition(transition)

    machine.start()
    event = Event("test")
    assert not machine.process_event(event)  # Should return False due to error

    assert recovery.called
    assert isinstance(recovery.last_error, ValueError)
    assert machine.current_state == state1  # Should remain in original state


def test_state_graph_integration():
    """Test that StateGraph is properly integrated."""
    state1 = State("state1")
    state2 = State("state2")
    machine = StateMachine(state1)
    machine._set_current_state(state1)  # Set current state before validation

    # Test graph state management
    machine.add_state(state2)
    assert state1 in machine.get_states()
    assert state2 in machine.get_states()

    # Test graph transition management
    transition = Transition(source=state1, target=state2)
    machine.add_transition(transition)
    assert transition in machine.get_transitions()


def test_composite_state_hierarchy():
    """Test hierarchical state structure with graph integration."""
    # Create states
    child1 = State("child1")
    child2 = State("child2")
    root = CompositeState("root")
    root._children = set()  # Initialize _children set

    # Create machine and build hierarchy
    machine = StateMachine(root)
    machine._set_current_state(root)  # Set current state before validation
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)

    # Set initial state through graph
    machine._graph.set_initial_state(root, child1)

    # Add transition
    transition = Transition(source=child1, target=child2)
    machine.add_transition(transition)

    # Start machine and verify hierarchy
    machine.start()
    assert machine.current_state == child1  # Should resolve to initial state
    assert machine._graph._parent_map[child1] == root
    assert machine._graph._parent_map[child2] == root

    # Test transition
    event = Event("test")
    machine.process_event(event)
    assert machine.current_state == child2


def test_machine_reset():
    """Test machine reset functionality."""
    state1 = State("state1")
    state2 = State("state2")
    machine = StateMachine(state1)
    machine._set_current_state(state1)  # Set current state before validation
    machine.add_state(state2)
    transition = Transition(source=state1, target=state2)
    machine.add_transition(transition)

    machine.start()
    event = Event("test")
    machine.process_event(event)
    assert machine.current_state == state2

    # Reset should restore to initial state
    machine.reset()
    assert machine.current_state == state1  # Should be back to initial state


def test_hook_behavior():
    """Test that hooks are properly called during state transitions."""
    state1 = State("state1")
    state2 = State("state2")

    hook = Mock()
    hook.on_enter = Mock()
    hook.on_exit = Mock()
    hook.on_error = Mock()

    machine = StateMachine(state1, hooks=[hook])
    machine._set_current_state(state1)  # Set current state before validation
    machine.add_state(state2)

    transition = Transition(source=state1, target=state2)
    machine.add_transition(transition)

    # Start machine and verify initial state hook
    machine.start()
    hook.on_enter.assert_called_once_with(state1)
    hook.on_exit.assert_not_called()  # Should not be called during start

    # Process event and verify transition hooks
    event = Event("test")
    machine.process_event(event)

    # Verify exit hook was called for state1
    assert hook.on_exit.call_count == 1
    hook.on_exit.assert_called_once_with(state1)

    # Verify enter hook was called for both states (initial and after transition)
    assert hook.on_enter.call_count == 2
    hook.on_enter.assert_has_calls([call(state1), call(state2)])

    # Verify error hook was not called
    hook.on_error.assert_not_called()


def test_composite_state_machine():
    """Test CompositeStateMachine with submachines."""
    # Create main machine states
    main_state1 = CompositeState("main_state1")
    main_state2 = State("main_state2")

    # Create submachine states
    sub_state1 = State("sub_state1")
    sub_state2 = State("sub_state2")

    # Create and setup submachine
    submachine = StateMachine(sub_state1)
    submachine._set_current_state(sub_state1)
    submachine.add_state(sub_state2)
    submachine.add_transition(Transition(source=sub_state1, target=sub_state2))

    # Create and setup main machine
    main_machine = CompositeStateMachine(main_state1)
    main_machine._set_current_state(main_state1)
    main_machine.add_state(main_state2)

    # Add submachine after all states and transitions are set up
    main_machine.add_submachine(main_state1, submachine)

    # Verify submachine integration
    assert sub_state1 in main_machine._graph.get_children(main_state1)
    assert sub_state2 in main_machine._graph.get_children(main_state1)


def test_history_state():
    """Test history state functionality."""
    # Create states
    root = CompositeState("root")
    root._children = set()  # Initialize _children set
    state1 = State("state1")
    state2 = State("state2")

    # Create machine and build hierarchy
    machine = StateMachine(root)
    machine._set_current_state(root)  # Set current state before validation
    machine.add_state(state1, parent=root)
    machine.add_state(state2, parent=root)

    # Set initial state through graph
    machine._graph.set_initial_state(root, state1)

    # Add transition
    machine.add_transition(Transition(source=state1, target=state2))

    # Start machine and make transition
    machine.start()
    assert machine.current_state == state1  # Should resolve to initial state
    machine.process_event(Event("test"))
    assert machine.current_state == state2

    # Stop machine to record history
    machine.stop()

    # Reset current state before restart
    machine._set_current_state(root)

    # Restart machine and verify history state is used
    machine.start()
    # The resolve_active_state should use the history state
    resolved_state = machine._graph.resolve_active_state(root)
    assert resolved_state == state2
    assert machine.current_state == state2  # Verify actual machine state


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
    assert graph.get_parent(child1) == cs
    assert graph.get_parent(child2) == cs
