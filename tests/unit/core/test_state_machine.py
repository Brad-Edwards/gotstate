# tests/unit/test_state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest
from unittest.mock import Mock, call

from hsm.core.states import State, CompositeState
from hsm.core.events import Event
from hsm.core.transitions import Transition
from hsm.core.state_machine import StateMachine
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
    
    machine = StateMachine(root)
    # Add states in correct order
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)
    
    # Add transition after all states are added
    transition = Transition(source=child1, target=child2)
    machine.add_transition(transition)
    
    # Verify hierarchy
    assert child1.parent == root
    assert child2.parent == root
    
    # Test transition within hierarchy
    machine.start()
    assert machine.get_current_state() == child1
    
    # Test transition
    event = Event("test")
    assert machine.process_event(event)
    assert machine.get_current_state() == child2

def test_composite_state_initial_state():
    """Test composite state with initial state."""
    child1 = State("child1")
    child2 = State("child2")
    root = CompositeState("root", initial_state=child1)
    
    machine = StateMachine(root)
    # Add states in correct order
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)
    
    # Verify initial state is set
    assert root._initial_state == child1
    
    # Start should enter initial state
    machine.start()
    assert machine.get_current_state() == child1
    
    # Test transition
    transition = Transition(source=child1, target=child2)
    machine.add_transition(transition)
    event = Event("test")
    assert machine.process_event(event)
    assert machine.get_current_state() == child2

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

def test_composite_history_integration(hook, validator):
    """Test composite states with history preservation."""
    # Create state hierarchy
    root = CompositeState("Root")
    group1 = CompositeState("Group1")
    group2 = CompositeState("Group2")

    # Create states for group1
    state1a = State("State1A")
    state1b = State("State1B")

    # Create states for group2
    state2a = State("State2A")
    state2b = State("State2B")

    # Create sub-machines
    sub_machine1 = StateMachine(initial_state=state1a, validator=validator, hooks=[hook])
    sub_machine1.add_state(state1a, parent=group1)
    sub_machine1.add_state(state1b, parent=group1)

    sub_machine2 = StateMachine(initial_state=state2a, validator=validator, hooks=[hook])
    sub_machine2.add_state(state2a, parent=group2)
    sub_machine2.add_state(state2b, parent=group2)

    # Add transitions
    t1 = Transition(source=state1a, target=state1b)
    t2 = Transition(source=state2a, target=state2b)
    t_groups = Transition(source=group1, target=group2)

    sub_machine1.add_transition(t1)
    sub_machine2.add_transition(t2)

    # Create composite machine
    c_machine = CompositeStateMachine(initial_state=root, validator=validator, hooks=[hook])
    
    # Add group1 and group2 as children of root
    c_machine.add_state(group1, parent=root)
    c_machine.add_state(group2, parent=root)
    
    c_machine.add_submachine(group1, sub_machine1)
    c_machine.add_submachine(group2, sub_machine2)
    c_machine.add_transition(t_groups)

    # Start machines
    c_machine.start()

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
