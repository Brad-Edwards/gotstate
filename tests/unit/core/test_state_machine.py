# tests/unit/test_state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import Mock, call

import pytest

from hsm.core.events import Event
from hsm.core.state_machine import CompositeStateMachine, StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import ValidationError


@pytest.fixture
def basic_machine():
    """Create a basic state machine setup."""
    state1 = State("state1")
    state2 = State("state2")
    machine = StateMachine(state1)
    machine.add_state(state2)
    transition = Transition(source=state1, target=state2)
    machine.add_transition(transition)
    return machine, state1, state2, transition


def test_initial_state(basic_machine):
    """Test that machine starts in the initial state."""
    machine, state1, _, _ = basic_machine
    assert machine.get_current_state() == state1


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
    assert machine.get_current_state() == state2


def test_no_valid_transition():
    """Test that invalid events are handled properly."""
    state = State("state")
    machine = StateMachine(state)
    machine.start()

    event = Event("unknown")
    assert not machine.process_event(event)  # Should return False
    assert machine.get_current_state() == state


def test_composite_state_hierarchy():
    """Test hierarchical state structure."""
    # Create child states first
    child1 = State("child1")
    child2 = State("child2")

    # Create root with child1 as initial state
    root = CompositeState("root", initial_state=child1)

    # Create machine with root as initial state
    machine = StateMachine(root)

    # Add states in correct order - parent first, then children
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)

    # Add transition after all states are added
    transition = Transition(source=child1, target=child2)
    machine.add_transition(transition)

    # Verify hierarchy
    assert child1.parent == root
    assert child2.parent == root

    machine.start()
    assert machine.current_state == root

    # Test transition
    event = Event("test")
    assert machine.process_event(event)
    assert machine.current_state == child2


def test_composite_state_initial_state():
    """Test composite state with initial state."""
    child1 = State("child1")
    child2 = State("child2")
    root = CompositeState("root", initial_state=child1)

    # Create machine with root as initial state
    machine = StateMachine(root)

    # Add states in correct order - parent first, then children
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)

    # Verify initial state is set
    assert root._initial_state == child1

    machine.start()
    assert machine.current_state == root

    # Test transition
    transition = Transition(source=child1, target=child2)
    machine.add_transition(transition)
    event = Event("test")
    assert machine.process_event(event)
    assert machine.current_state == child2


def test_invalid_machine_structure():
    """Test validation of invalid machine structures."""
    composite = CompositeState("composite")
    machine = StateMachine(composite)

    # Composite state with no children should fail validation
    with pytest.raises(ValidationError) as exc:
        machine.start()
    assert "no children" in str(exc.value)


def test_nested_composite_states():
    """Test nested composite state structure."""
    inner_initial = State("inner_initial")
    inner = CompositeState("inner", initial_state=inner_initial)
    outer = CompositeState("outer", initial_state=inner)

    machine = StateMachine(outer)
    machine.add_state(inner, parent=outer)
    machine.add_state(inner_initial, parent=inner)

    # Verify hierarchy
    assert inner.parent == outer
    assert inner_initial.parent == inner

    # Verify initial states
    assert outer._initial_state == inner
    assert inner._initial_state == inner_initial


def test_hook_behavior():
    """Test that hooks are properly called during state transitions."""
    state1 = State("state1")
    state2 = State("state2")

    # Create a mock hook
    hook = Mock()
    hook.on_enter = Mock()
    hook.on_exit = Mock()

    # Create machine with hook
    machine = StateMachine(state1, hooks=[hook])
    machine.add_state(state2)

    # Add transition
    transition = Transition(source=state1, target=state2)
    machine.add_transition(transition)

    # Start machine and verify initial state hook
    machine.start()
    hook.on_enter.assert_called_once_with(state1)

    # Process event and verify transition hooks
    event = Event("test")
    machine.process_event(event)

    # Verify exit from state1 and enter to state2
    hook.on_exit.assert_called_once_with(state1)
    assert hook.on_enter.call_count == 2  # Once for initial, once for transition
    hook.on_enter.assert_has_calls([call(state1), call(state2)])
