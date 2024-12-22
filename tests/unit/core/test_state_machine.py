# tests/unit/test_state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import Mock, call, patch

import pytest

from gotstate.core.events import Event
from gotstate.core.state_machine import CompositeStateMachine, StateMachine, _ErrorRecoveryStrategy
from gotstate.core.states import CompositeState, State
from gotstate.core.transitions import Transition
from gotstate.core.validations import ValidationError
from gotstate.runtime.graph import StateGraph


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


def test_start_already_started():
    """Test starting an already started machine."""
    state = State("test")
    machine = StateMachine(state)
    machine._set_current_state(state)
    machine.start()
    assert machine._started
    # Second start should be a no-op
    machine.start()
    assert machine._started


def test_stop_not_started():
    """Test stopping a machine that wasn't started."""
    state = State("test")
    machine = StateMachine(state)
    machine.stop()  # Should not raise
    assert not machine._started


def test_transition_with_no_current_state():
    """Test transition attempt with no current state."""
    state1 = State("state1")
    state2 = State("state2")
    machine = StateMachine(state1)
    machine.add_state(state2)
    machine.add_transition(Transition(state1, state2))

    # Don't set current state or start machine
    event = Event("test")
    assert not machine.process_event(event)


def test_composite_state_initial_transition():
    """Test initial transition into composite state."""
    root = CompositeState("root")
    composite = CompositeState("composite")
    leaf = State("leaf")

    machine = StateMachine(root)
    machine.add_state(composite, parent=root)
    machine.add_state(leaf, parent=composite)

    machine._graph.set_initial_state(root, composite)
    machine._graph.set_initial_state(composite, leaf)

    machine.start()
    assert machine.current_state == leaf


def test_error_in_state_hooks():
    """Test error handling in state enter/exit hooks."""
    state1 = State("state1")
    state2 = State("state2")

    def failing_enter():
        raise ValueError("Enter failed")

    state2.on_enter = failing_enter

    machine = StateMachine(state1)
    machine._set_current_state(state1)
    machine.add_state(state2)
    machine.add_transition(Transition(state1, state2))

    machine.start()
    with pytest.raises(ValueError, match="Enter failed"):
        machine.process_event(Event("test"))

    # Machine should remain in original state
    assert machine.current_state == state1


def test_async_hook_skipping():
    """Test that async hooks are skipped in sync context."""
    state = State("test")

    async def async_enter(state):
        pass  # This should be skipped

    hook = Mock()
    hook.on_enter = async_enter

    machine = StateMachine(state, hooks=[hook])
    machine._set_current_state(state)
    machine.start()  # Should not raise


def test_reset_not_started():
    """Test reset on a machine that wasn't started."""
    state = State("test")
    machine = StateMachine(state)
    machine.reset()  # Should not raise
    assert not machine._started


def test_composite_state_machine_validation():
    """Test validation in composite state machine."""
    root = CompositeState("root")
    composite = CompositeState("composite")  # No initial state

    machine = CompositeStateMachine(root)
    machine.add_state(composite, parent=root)
    machine._graph.set_initial_state(root, composite)

    with pytest.raises(ValidationError, match="has no initial state"):
        machine.start()


def test_composite_state_machine_stop_no_current():
    """Test stopping composite machine with no current state."""
    root = CompositeState("root")
    machine = CompositeStateMachine(root)
    machine._started = True
    machine.stop()  # Should not raise
    assert not machine._started


def test_start_with_history_state():
    """Test starting machine with history state."""
    root = CompositeState("root")
    composite = CompositeState("composite")
    state1 = State("state1")
    state2 = State("state2")

    machine = StateMachine(root)
    machine.add_state(composite, parent=root)
    machine.add_state(state1, parent=composite)
    machine.add_state(state2, parent=composite)

    machine._graph.set_initial_state(root, composite)
    machine._graph.set_initial_state(composite, state1)

    # Record history state
    machine._graph.record_history(composite, state2)

    machine.start()
    # Should use history state instead of initial state
    assert machine.current_state == state2


def test_start_with_deep_history():
    """Test starting machine with deep history states."""
    root = CompositeState("root")
    comp1 = CompositeState("comp1")
    comp2 = CompositeState("comp2")
    state1 = State("state1")
    state2 = State("state2")

    machine = StateMachine(root)
    machine.add_state(comp1, parent=root)
    machine.add_state(comp2, parent=comp1)
    machine.add_state(state1, parent=comp2)
    machine.add_state(state2, parent=comp2)

    machine._graph.set_initial_state(root, comp1)
    machine._graph.set_initial_state(comp1, comp2)
    machine._graph.set_initial_state(comp2, state1)

    # Start machine first to establish state
    machine.start()
    assert machine.current_state == state1

    # Now transition to state2 and stop to record history
    machine._set_current_state(state2)
    machine.stop()

    # Restart and check history state
    machine.start()
    assert machine.current_state == state2


def test_start_validation_error():
    """Test start with validation error."""
    root = CompositeState("root")
    state = State("state")

    machine = StateMachine(root)
    machine.add_state(state, parent=root)
    # Don't set initial state for composite state

    # First start to set up the machine
    machine.start()
    machine.stop()

    # Now clear the initial state to trigger validation error
    machine._graph._initial_states[root] = None

    with pytest.raises(ValidationError, match="Composite state 'root' has no initial state"):
        machine.start()


def test_start_with_no_history_or_initial():
    """Test starting machine when no history or initial state is found."""
    root = State("root")  # Changed to non-composite state
    machine = StateMachine(root)
    # Clear any initial states
    machine._graph._initial_states.clear()

    with pytest.raises(ValidationError, match="StateMachine must have an initial state"):
        machine.start()


def test_start_with_resolve_active_state():
    """Test starting machine using resolve_active_state."""
    root = CompositeState("root")
    comp = CompositeState("comp")
    state = State("state")

    machine = StateMachine(root)
    machine.add_state(comp, parent=root)
    machine.add_state(state, parent=comp)

    machine._graph.set_initial_state(root, comp)
    machine._graph.set_initial_state(comp, state)

    # Mock resolve_active_state to return specific state
    original_resolve = machine._graph.resolve_active_state
    machine._graph.resolve_active_state = Mock(return_value=state)

    machine.start()
    assert machine.current_state == state

    # Restore original method
    machine._graph.resolve_active_state = original_resolve


def test_start_history_resolution():
    """Test history state resolution during start."""
    root = CompositeState("root")
    comp1 = CompositeState("comp1")
    comp2 = CompositeState("comp2")
    leaf1 = State("leaf1")
    leaf2 = State("leaf2")

    machine = StateMachine(root)
    machine.add_state(comp1, parent=root)
    machine.add_state(comp2, parent=comp1)
    machine.add_state(leaf1, parent=comp2)
    machine.add_state(leaf2, parent=comp2)

    # Set up initial states
    machine._graph.set_initial_state(root, comp1)
    machine._graph.set_initial_state(comp1, comp2)
    machine._graph.set_initial_state(comp2, leaf1)

    # Start machine and transition to leaf2
    machine.start()
    machine._set_current_state(leaf2)

    # Record history at different levels
    machine._graph.record_history(comp2, leaf2)
    machine._graph.record_history(comp1, comp2)

    # Stop and restart to test history resolution
    machine.stop()
    machine.start()

    assert machine.current_state == leaf2


def test_composite_transition_with_initial_states():
    """Test transitions into composite states with initial states."""
    root = CompositeState("root")
    comp1 = CompositeState("comp1")
    comp2 = CompositeState("comp2")
    leaf1 = State("leaf1")
    leaf2 = State("leaf2")

    machine = StateMachine(root)
    machine.add_state(comp1, parent=root)
    machine.add_state(comp2, parent=root)
    machine.add_state(leaf1, parent=comp1)
    machine.add_state(leaf2, parent=comp2)

    machine._graph.set_initial_state(root, comp1)
    machine._graph.set_initial_state(comp1, leaf1)
    machine._graph.set_initial_state(comp2, leaf2)

    # Add transition between leaf states instead of composite states
    transition = Transition(source=leaf1, target=comp2)
    machine.add_transition(transition)

    machine.start()
    assert machine.current_state == leaf1

    # Transition should go to comp2's initial state
    machine.process_event(Event("test"))
    assert machine.current_state == leaf2


def test_parent_state_guard_evaluation():
    """Test guard evaluation in parent state transitions."""
    root = CompositeState("root")
    comp = CompositeState("comp")
    leaf1 = State("leaf1")
    leaf2 = State("leaf2")

    machine = StateMachine(root)
    machine.add_state(comp, parent=root)
    machine.add_state(leaf1, parent=comp)
    machine.add_state(leaf2, parent=root)  # Move leaf2 to root level

    # Add transition from leaf1 to leaf2 with guard
    def guard(event):
        return event.name == "test"

    transition = Transition(source=leaf1, target=leaf2, guards=[guard])
    machine.add_transition(transition)

    machine._graph.set_initial_state(root, comp)
    machine._graph.set_initial_state(comp, leaf1)

    machine.start()
    assert machine.current_state == leaf1

    # Test guard evaluation
    assert not machine.process_event(Event("wrong"))
    assert machine.process_event(Event("test"))
    assert machine.current_state == leaf2


def test_stop_with_history_recording():
    """Test history recording during stop."""
    root = CompositeState("root")
    comp1 = CompositeState("comp1")
    comp2 = CompositeState("comp2")
    leaf = State("leaf")

    machine = StateMachine(root)
    machine.add_state(comp1, parent=root)
    machine.add_state(comp2, parent=comp1)
    machine.add_state(leaf, parent=comp2)

    machine._graph.set_initial_state(root, comp1)
    machine._graph.set_initial_state(comp1, comp2)
    machine._graph.set_initial_state(comp2, leaf)

    machine.start()
    assert machine.current_state == leaf

    # Stop should record history for all composite ancestors
    machine.stop()

    # Verify history was recorded - history is recorded as the leaf state for all ancestors
    assert machine._graph.get_history_state(comp2) == leaf
    assert machine._graph.get_history_state(comp1) == leaf
    assert machine._graph.get_history_state(root) == leaf


def test_state_machine_validation_errors():
    """Test validation error handling in state machine."""
    # Create composite with no children or initial state
    composite = CompositeState("composite")
    composite._children = set()

    machine = StateMachine(composite)

    errors = machine.validate()
    assert len(errors) > 0
    assert any("no children" in error.lower() for error in errors)


def test_state_machine_hook_order():
    """Test hook execution order during transitions."""
    initial = State("initial")
    target = State("target")

    # Track execution order
    order = []

    class OrderTrackingHook:
        def on_exit(self, state):
            order.append(f"exit_{state.name}")

        def on_enter(self, state):
            order.append(f"enter_{state.name}")

        def on_transition(self, source, target):
            order.append(f"transition_{source.name}_to_{target.name}")

    machine = StateMachine(initial, hooks=[OrderTrackingHook()])
    machine.add_state(target)
    machine.add_transition(Transition(initial, target))

    machine.start()
    machine.process_event(Event("test"))

    # Verify correct order
    expected = [
        "enter_initial",  # From start()
        "exit_initial",  # From transition
        "transition_initial_to_target",
        "enter_target",
    ]
    assert order == expected


def test_state_machine_data_management():
    """Test state data management."""
    state = State("test")
    machine = StateMachine(state)
    machine._set_current_state(state)  # Need to set current state

    # Set data through graph
    machine._graph.set_state_data(state, "key", "value")
    assert machine._graph.get_state_data(state)["key"] == "value"

    # Test non-existent state
    non_existent = State("non_existent")
    with pytest.raises(KeyError):
        machine._graph.set_state_data(non_existent, "key", "value")

    with pytest.raises(AttributeError, match="State data cannot be accessed directly"):
        machine._graph.get_state_data(non_existent)


def test_state_machine_transition_error_handling():
    """Test error handling during transitions."""
    initial = State("initial")
    target = State("target")

    def failing_action(event):
        raise ValueError("Action failed")

    machine = StateMachine(initial)
    machine._set_current_state(initial)  # Need to set current state
    machine.add_state(target)
    machine.add_transition(
        Transition(
            source=initial, target=target, actions=[failing_action]  # Use action instead of guard for error testing
        )
    )

    machine.start()
    with pytest.raises(ValueError, match="Action failed"):
        machine.process_event(Event("test"))

    # Machine should remain in initial state
    assert machine.current_state == initial
