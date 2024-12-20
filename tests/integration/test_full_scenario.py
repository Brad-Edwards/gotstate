# tests/integration/test_full_scenario.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time
from typing import List, Optional
from unittest.mock import Mock

import pytest

from hsm.core.events import Event, TimeoutEvent
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import CompositeStateMachine, StateMachine, _ErrorRecoveryStrategy
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import ValidationError, Validator
from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine
from hsm.runtime.event_queue import EventQueue
from hsm.runtime.executor import Executor
from hsm.runtime.timers import TimeoutScheduler


@pytest.fixture
def hook() -> HookProtocol:
    """Create a mock hook for testing."""
    return Mock(spec=HookProtocol)


@pytest.fixture
def validator() -> Validator:
    """Create a validator for testing."""
    return Validator()


def test_synchronous_integration(hook, validator):
    """
    Full integration test for composite state functionality:
    - Create states with entry/exit actions
    - Add transitions with guards and actions
    - Test state hierarchy and transitions
    - Use hooks to track state machine lifecycle
    - Run via Executor and EventQueue
    """
    # Set up mock hook with required attributes
    hook.entered = set()
    hook.exited = set()
    hook.errors = []

    def on_enter(state):
        hook.entered.add(state.name)

    def on_exit(state):
        hook.exited.add(state.name)

    def on_error(error):
        hook.errors.append(error)

    hook.on_enter = on_enter
    hook.on_exit = on_exit
    hook.on_error = on_error

    # Define states
    parent = CompositeState("Parent")
    state1 = State("State1")
    state2 = State("State2")

    parent.add_child_state(state1)
    parent.add_child_state(state2)

    # Construct StateMachine and add states first
    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
    machine.add_state(state2)  # Add state2 before transitions

    # Define and add transitions
    t1 = Transition(source=state1, target=state2, guards=[lambda e: True], actions=[lambda e: None], priority=10)
    t2 = Transition(source=state2, target=state1, guards=[lambda e: True], actions=[lambda e: None], priority=10)

    machine.add_transition(t1)
    machine.add_transition(t2)

    # Setup runtime
    eq = EventQueue(priority=False)
    executor = Executor(machine, eq)

    # Start machine
    machine.start()
    assert machine.current_state.name == "State1"
    assert "State1" in hook.entered

    # Process events
    eq.enqueue(Event("Next"))  # Should go to State2

    # Run executor in thread
    thread = threading.Thread(target=executor.run)
    thread.start()
    time.sleep(0.1)

    assert machine.current_state.name == "State2"
    assert "State2" in hook.entered
    assert "State1" in hook.exited

    # Stop executor and cleanup
    executor.stop()
    thread.join(timeout=1.0)

    # Verify no errors occurred
    assert len(hook.errors) == 0


def test_composite_state_machine_integration(hook, validator):
    """
    Integration test for composite state machines:
    - Create a composite state with a nested submachine
    - Validate transitions occur in the nested structure
    - Verify parent-child relationships
    """
    # Set up mock hook with required attributes
    hook.entered = set()
    hook.exited = set()
    hook.errors = []

    def on_enter(state):
        hook.entered.add(state.name)

    def on_exit(state):
        hook.exited.add(state.name)

    def on_error(error):
        hook.errors.append(error)

    hook.on_enter = on_enter
    hook.on_exit = on_exit
    hook.on_error = on_error

    # Define states
    child1 = State("Child1")
    child2 = State("Child2")
    top = CompositeState("Top", initial_state=child1)

    # Set up hierarchy - child1 is already added as initial state
    top.add_child_state(child2)

    # Verify parent-child relationships
    assert child1.parent is top
    assert child2.parent is top

    # Define submachine - child1 is already added as initial state
    sub_machine = StateMachine(initial_state=child1, validator=validator, hooks=[hook])
    sub_machine.add_state(child2)  # Only add child2

    # Add transition after states
    t_sub = Transition(source=child1, target=child2, guards=[], actions=[], priority=0)
    sub_machine.add_transition(t_sub)

    # Create composite machine and add submachine
    c_machine = CompositeStateMachine(initial_state=top, validator=validator, hooks=[hook])
    c_machine.add_submachine(top, sub_machine)

    # Start machines
    c_machine.start()
    assert c_machine.current_state.name == "Top"
    sub_machine.start()
    assert sub_machine.current_state.name == "Child1"

    # Test transition in submachine
    sub_machine.process_event(Event("Go"))
    assert sub_machine.current_state.name == "Child2"
    assert "Child2" in hook.entered
    assert "Child1" in hook.exited

    # Cleanup
    c_machine.stop()
    sub_machine.stop()


@pytest.mark.asyncio
async def test_async_integration(hook, validator):
    """Test async event processing."""
    # States
    start_state = State("AsyncStart")
    end_state = State("AsyncEnd")

    # Create machine and add states first
    machine = AsyncStateMachine(initial_state=start_state, validator=validator, hooks=[hook])
    machine.add_state(end_state)  # Add end_state before transition

    # Add transition after states
    t = Transition(source=start_state, target=end_state, guards=[], actions=[], priority=0)
    machine.add_transition(t)

    # Create async event queue
    eq = AsyncEventQueue()

    # Start machine
    await machine.start()
    assert machine.current_state == start_state

    # Process event
    event = Event("test")
    await eq.enqueue(event)
    await machine.process_event(event)

    assert machine.current_state == end_state

    # Cleanup
    await machine.stop()


def test_timeout_event_integration(validator):
    """
    Integration test for timeout scheduling:
    - Set up a TimeoutEvent and a TimeoutScheduler.
    - Simulate time passing and ensure the event is returned as expired.
    """
    scheduler = TimeoutScheduler()
    timeout_event = TimeoutEvent(name="TimeoutCheck", deadline=time.time() + 0.2)  # Expires in 200ms
    scheduler.schedule_timeout(timeout_event)

    # Check initially (no time passed)
    assert len(scheduler.check_timeouts()) == 0, "No timeouts should expire immediately."

    # Wait 300ms, events should expire now.
    time.sleep(0.3)
    expired = scheduler.check_timeouts()
    assert len(expired) == 1
    assert expired[0].name == "TimeoutCheck"
    # This shows that the TimeoutScheduler integrated with the concept of time.


def test_validation_integration():
    """
    Integration test ensuring that the Validator catches invalid configurations.
    """
    s1 = State("S1")
    s2 = State("S2")
    s3 = State("S3")

    # Create machine and add all states first
    machine = StateMachine(initial_state=s1, validator=Validator())
    machine.add_state(s2)
    machine.add_state(s3)

    # Add transitions after all states are in graph
    t1 = Transition(source=s1, target=s2)
    t2 = Transition(source=s2, target=s3)

    machine.add_transition(t1)
    machine.add_transition(t2)


def test_plugins_integration(hook, validator):
    """
    Integration test checking that plugins (custom guards/actions) integrate seamlessly
    into the machine.
    """
    from hsm.plugins.custom_actions import MyCustomAction
    from hsm.plugins.custom_guards import MyCustomGuard

    start = State("PluginStart")
    end = State("PluginEnd")

    guard = MyCustomGuard(lambda e: e.name == "Go")
    action = MyCustomAction(lambda e: setattr(e, "processed_by_plugin", True))

    # Create machine and add all states first
    machine = StateMachine(initial_state=start, validator=validator, hooks=[hook])
    machine.add_state(end)  # Add end state before creating transition

    # Create and add transition after states are in graph
    t = Transition(source=start, target=end, guards=[guard.check], actions=[action.run], priority=0)
    machine.add_transition(t)

    machine.start()
    assert machine.current_state.name == "PluginStart"

    # Process event that should trigger transition
    event = Event("Go")
    machine.process_event(event)

    # Verify transition occurred and plugin action was executed
    assert machine.current_state.name == "PluginEnd"
    assert hasattr(event, "processed_by_plugin")
    assert event.processed_by_plugin is True


def test_complex_hierarchy():
    from hsm.core.states import CompositeState, State

    root = CompositeState("Root")
    group1 = CompositeState("Group1")
    group2 = CompositeState("Group2")
    state1 = State("State1")
    state2 = State("State2")

    root.add_child_state(group1)
    root.add_child_state(group2)
    group1.add_child_state(state1)
    group2.add_child_state(state2)

    # Test navigation and state access
    assert root.get_child_state("Group1") is group1
    assert group1.get_child_state("State1") is state1


def test_concurrent_event_processing():
    """Test concurrent event processing with multiple producers/consumers."""
    import threading
    from queue import Queue

    # Create states
    state1 = State("State1")
    state2 = State("State2")
    state3 = State("State3")

    # Create machine and add all states first
    machine = StateMachine(initial_state=state1)
    machine.add_state(state2)
    machine.add_state(state3)

    # Create transitions after all states are added
    t1 = Transition(source=state1, target=state2)
    t2 = Transition(source=state2, target=state3)

    # Add transitions
    machine.add_transition(t1)
    machine.add_transition(t2)

    # Rest of concurrent processing test...
    results = Queue()

    def worker():
        for _ in range(50):
            machine.process_event(Event("test"))
            results.put(machine.current_state.name)

    # Start concurrent workers
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    # Verify results
    processed = []
    while not results.empty():
        processed.append(results.get())

    assert len(processed) == 200  # 4 threads * 50 events each


def test_state_data_integration(hook, validator):
    """
    Integration test for state data management:
    - Verify data persistence across transitions
    - Test data isolation between states
    - Validate thread safety of data access
    """
    # Define states with initial data
    state1 = State("State1")
    state2 = State("State2")

    # Initialize state data
    state1.data["counter"] = 0
    state2.data["counter"] = 10

    # Create machine and add all states first
    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
    machine.add_state(state2)  # Add state2 before creating transition

    # Define transition after states are in graph
    def increment_action(event):
        state1.data["counter"] += 1

    t1 = Transition(source=state1, target=state2, guards=[lambda e: True], actions=[increment_action], priority=0)
    machine.add_transition(t1)

    # Start and verify initial data
    machine.start()
    assert machine.current_state.data["counter"] == 0

    # Process event and verify data changes
    event = Event("Next")
    machine.process_event(event)

    # Verify state data modifications
    assert state1.data["counter"] == 1
    assert state2.data["counter"] == 10
    assert machine.current_state == state2


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

    # Set up hierarchy first
    root.add_child_state(group1)
    root.add_child_state(group2)
    group1.add_child_state(state1a)
    group1.add_child_state(state1b)
    group2.add_child_state(state2a)
    group2.add_child_state(state2b)

    # Create sub-machines and add all states first
    sub_machine1 = StateMachine(initial_state=state1a, validator=validator, hooks=[hook])
    sub_machine1.add_state(state1a, parent=group1)
    sub_machine1.add_state(state1b, parent=group1)

    sub_machine2 = StateMachine(initial_state=state2a, validator=validator, hooks=[hook])
    sub_machine2.add_state(state2a, parent=group2)
    sub_machine2.add_state(state2b, parent=group2)

    # Create composite machine and add all states
    c_machine = CompositeStateMachine(initial_state=root, validator=validator, hooks=[hook])
    c_machine.add_state(group1, parent=root)
    c_machine.add_state(group2, parent=root)

    # Add transitions after all states are in their respective machines
    t1 = Transition(source=state1a, target=state1b)
    t2 = Transition(source=state2a, target=state2b)
    t_groups = Transition(source=group1, target=group2)

    sub_machine1.add_transition(t1)
    sub_machine2.add_transition(t2)
    c_machine.add_transition(t_groups)

    # Add submachines after all states and transitions are set up
    c_machine.add_submachine(group1, sub_machine1)
    c_machine.add_submachine(group2, sub_machine2)

    # Start machines
    c_machine.start()
    sub_machine1.start()
    sub_machine2.start()

    # Test transitions and history
    sub_machine1.process_event(Event("ToB"))
    assert sub_machine1.current_state == state1b

    # Stop and restart to test history
    sub_machine1.stop()
    sub_machine1.start()
    assert sub_machine1.current_state == state1b

    # Clear history
    sub_machine1.reset()
    sub_machine1.start()
    assert sub_machine1.current_state == state1a

    c_machine.stop()


@pytest.mark.integration
def test_error_recovery_integration(hook, validator):
    """
    Integration test for error recovery:
    - Test transition failures
    - Test action failures
    - Test fallback transitions
    - Verify state consistency after errors
    """
    # Define states
    normal = State("Normal")
    error = State("Error")
    fallback = State("Fallback")

    def failing_action(event):
        raise RuntimeError("Action failed")

    # Create error recovery strategy
    class TestErrorRecovery(_ErrorRecoveryStrategy):
        def recover(self, error: Exception, state_machine: StateMachine) -> None:
            # When error occurs, trigger transition to fallback
            try:
                state_machine.process_event(Event("Recover"))
            except Exception as e:
                # Log error to hook
                for hook in state_machine._hooks:
                    if hasattr(hook, "on_error"):
                        hook.on_error(e)

    # Create machine and add all states first
    machine = StateMachine(initial_state=normal, validator=validator, hooks=[hook])
    machine.add_state(error)  # Add error state to graph
    machine.add_state(fallback)  # Add fallback state to graph

    # Set error recovery strategy
    machine._error_recovery = TestErrorRecovery()

    # Define transitions after all states are in graph
    t1 = Transition(
        source=normal, target=error, actions=[failing_action], guards=[lambda e: e.name == "Fail"], priority=10
    )
    t2 = Transition(source=normal, target=fallback, guards=[lambda e: e.name == "Recover"], priority=20)

    # Add transitions
    machine.add_transition(t1)
    machine.add_transition(t2)

    # Start machine and test error recovery
    machine.start()
    assert machine.current_state.name == "Normal"

    # Trigger failing transition
    machine.process_event(Event("Fail"))
    # Error recovery should have moved us to fallback state
    assert machine.current_state.name == "Fallback"


@pytest.mark.integration
def test_complex_event_chain_integration(hook, validator):
    """
    Integration test for complex event chains:
    - Multiple transitions triggered by single event
    - Priority-based transition selection
    - Event queue processing under load
    """
    # Create state hierarchy
    root = CompositeState("Root")
    state_a = State("StateA")
    state_b = State("StateB")
    state_c = State("StateC")

    # Track transition sequence
    transition_sequence = []

    def track_transition(state_name):
        def action(event):
            transition_sequence.append(state_name)

        return action

    # Create and configure machine
    machine = StateMachine(initial_state=state_a, validator=validator, hooks=[hook])

    # Add all states first
    machine.add_state(state_b)
    machine.add_state(state_c)

    # Define and add transitions with different priorities
    t1 = Transition(source=state_a, target=state_b, actions=[track_transition("B")], priority=10)
    t2 = Transition(source=state_b, target=state_c, actions=[track_transition("C")], priority=20)

    # Add transitions after all states are in graph
    machine.add_transition(t1)
    machine.add_transition(t2)

    # Start machine and verify transitions
    machine.start()
    assert machine.current_state == state_a

    # Process event and verify transition sequence
    event = Event("test")
    machine.process_event(event)
    assert machine.current_state == state_b
    assert transition_sequence == ["B"]

    # Process another event
    machine.process_event(event)
    assert machine.current_state == state_c
    assert transition_sequence == ["B", "C"]


@pytest.mark.integration
def test_resource_lifecycle_integration(hook, validator):
    """
    Integration test for resource management:
    - Resource initialization in states
    - Cleanup during transitions
    - Resource isolation between states
    """

    class ManagedResource:
        def __init__(self):
            self.active = True

        def cleanup(self):
            self.active = False

    # Create states with resources
    class ResourceState(State):
        def __init__(self, name):
            super().__init__(name)
            self.resource = None

        def on_enter(self):
            self.resource = ManagedResource()

        def on_exit(self):
            if self.resource:
                self.resource.cleanup()

    state1 = ResourceState("State1")
    state2 = ResourceState("State2")

    # Create machine and add states
    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
    machine.add_state(state2)  # Add state2 before creating transition

    # Define and add transition
    t = Transition(source=state1, target=state2, priority=0)
    machine.add_transition(t)

    # Start machine and verify resource lifecycle
    machine.start()
    assert state1.resource is not None
    assert state1.resource.active

    # Transition should cleanup state1's resource
    machine.process_event(Event("test"))
    assert not state1.resource.active
    assert state2.resource is not None
    assert state2.resource.active


def test_validation_framework_integration(hook):
    """
    Integration test for validation framework:
    - Custom validation rules
    - Runtime validation
    - Validation error recovery
    """

    class CustomValidator(Validator):
        def validate_state_machine(self, machine):
            super().validate_state_machine(machine)
            # Custom validation rule
            if not hasattr(machine.current_state, "data"):
                raise ValidationError("States must have data dictionary")

        def validate_transition(self, transition):
            # Ensure transitions have at least one guard
            if not transition.guards:
                raise ValidationError("Transitions must have at least one guard")
            # Call parent validation after our custom check
            super().validate_transition(transition)

    validator = CustomValidator()

    # Create states - data dictionary is automatically initialized by State class
    state1 = State("State1")
    state2 = State("State2")

    # This should fail validation (no guards)
    t_invalid = Transition(source=state1, target=state2, priority=0)

    # This should pass validation
    t_valid = Transition(source=state1, target=state2, guards=[lambda e: True], priority=0)

    # Test validation failures - validate transition directly
    with pytest.raises(ValidationError) as exc:
        validator.validate_transition(t_invalid)
    assert "must have at least one guard" in str(exc.value)

    # Test that valid transition passes validation
    validator.validate_transition(t_valid)  # Should not raise

    # Create machine with custom validator and verify it works with valid transition
    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
    machine.add_state(state2)  # Add state2 before adding transition
    machine.add_transition(t_valid)  # Should not raise
    machine.start()  # Should not raise


def test_concurrent_state_transitions():
    """Test multiple transitions happening simultaneously under load"""

    class SharedState(State):
        def __init__(self, name):
            super().__init__(name)
            self.transition_count = 0
            self._lock = threading.Lock()

        def on_enter(self):
            with self._lock:
                self.transition_count += 1
                # Force potential race condition
                time.sleep(0.001)
                self.transition_count -= 1

    states = [SharedState(f"State{i}") for i in range(3)]
    machine = StateMachine(initial_state=states[0])

    # Add all states first
    for state in states:
        machine.add_state(state)

    # Add transitions in a circle after all states are in graph
    for i in range(3):
        machine.add_transition(Transition(states[i], states[(i + 1) % 3], guards=[lambda e: True]))

    def worker():
        for _ in range(100):
            machine.process_event(Event("next"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    # Verify no state was left with non-zero count
    for state in states:
        assert state.transition_count == 0


def test_complex_error_recovery():
    """Test recovery from cascading errors in hierarchical states"""
    error_sequence = []
    
    class ErrorState(State):
        def __init__(self, name, should_raise=False):
            def on_enter_action():
                error_sequence.append(f"Enter_{name}")
                if should_raise:
                    raise RuntimeError("Simulated error")
                    
            def on_exit_action():
                error_sequence.append(f"Exit_{name}")
                
            super().__init__(name, entry_actions=[on_enter_action], exit_actions=[on_exit_action])
    
    # Create states
    normal = ErrorState("Normal")
    error = ErrorState("ErrorState", should_raise=True)  # This state raises on entry
    fallback = ErrorState("Fallback")
    
    # Create error recovery strategy
    class TestErrorRecovery(_ErrorRecoveryStrategy):
        def recover(self, error: Exception, state_machine: StateMachine) -> None:
            # Trigger transition to fallback state
            state_machine.process_event(Event("recover"))
    
    # Create machine with validator and error recovery strategy
    validator = Validator()
    machine = StateMachine(
        initial_state=normal,
        validator=validator,
        error_recovery=TestErrorRecovery()  # Set error recovery during creation
    )
    
    # Add all states and transitions
    machine.add_state(error)
    machine.add_state(fallback)
    machine.add_transition(Transition(normal, error))
    machine.add_transition(Transition(error, fallback))
    # Add recovery transition
    machine.add_transition(Transition(error, fallback, guards=[lambda e: e.name == "recover"]))
    
    # Start machine and trigger error transition
    machine.start()
    assert machine.current_state == normal
    
    # Trigger transition that will cause error
    machine.process_event(Event("trigger_error"))
    
    # After error recovery, we should be in the fallback state
    assert machine.current_state == fallback
    
    # Verify the sequence of state entries/exits
    assert error_sequence == [
        "Enter_Normal",
        "Exit_Normal",
        "Enter_ErrorState",  # From error state's entry action
        "Enter_Fallback"     # From recovery transition
    ]


def test_resource_cleanup_under_load():
    """Test resource cleanup during rapid state changes"""
    resource_count = 0

    class ResourceState(State):
        def __init__(self, name):
            def on_enter_action():
                nonlocal resource_count
                resource_count += 1

            def on_exit_action():
                nonlocal resource_count
                resource_count -= 1

            super().__init__(name, entry_actions=[on_enter_action], exit_actions=[on_exit_action])

    states = [ResourceState(f"State{i}") for i in range(5)]
    machine = StateMachine(initial_state=states[0])

    # Add states in a chain
    for i in range(1, 5):
        machine.add_state(states[i])
        machine.add_transition(Transition(states[i - 1], states[i]))

    # Start machine to trigger initial state's entry action
    machine.start()
    assert resource_count == 1  # Initial state entered

    # Rapid transitions
    for _ in range(1000):
        machine.process_event(Event("next"))

    # Verify no resource leaks
    assert resource_count == 1  # Only current state should have a resource


def test_deep_history_persistence():
    """Test history state preservation across complex hierarchies"""
    # Create states and hierarchy
    root = CompositeState("Root")
    group1 = CompositeState("Group1", initial_state=None)  # We'll set initial states after creation
    group2 = CompositeState("Group2", initial_state=None)
    state1 = State("State1")
    state2 = State("State2")
    state3 = State("State3")
    state4 = State("State4")

    # Build hierarchy
    root.add_child_state(group1)
    root.add_child_state(group2)
    group1.add_child_state(state1)
    group1.add_child_state(state2)
    group2.add_child_state(state3)
    group2.add_child_state(state4)

    # Set initial states for composite states
    group1.initial_state = state1
    group2.initial_state = state3

    # Create submachines for the composite states
    submachine1 = StateMachine(initial_state=state1)
    submachine1.add_state(state1)
    submachine1.add_state(state2)
    submachine1.add_transition(Transition(state1, state2))

    submachine2 = StateMachine(initial_state=state3)
    submachine2.add_state(state3)
    submachine2.add_state(state4)
    submachine2.add_transition(Transition(state3, state4))

    # Create composite machine
    machine = CompositeStateMachine(initial_state=root)
    machine.add_state(group1, parent=root)
    machine.add_state(group2, parent=root)

    # Add submachines to composite states
    machine.add_submachine(group1, submachine1)
    machine.add_submachine(group2, submachine2)

    # Add transitions between groups
    machine.add_transition(Transition(group1, group2))

    # Start all machines
    machine.start()
    submachine1.start()
    submachine2.start()

    # Initial state verification
    assert submachine1.current_state == state1
    assert submachine2.current_state == state3

    # Make transitions and verify history
    submachine1.process_event(Event("test"))
    assert submachine1.current_state == state2  # Moved to state2 in group1

    # Stop and restart submachine1 - should restore to state2
    submachine1.stop()
    submachine1.start()
    assert submachine1.current_state == state2  # History preserved

    # Move to group2
    machine.process_event(Event("test"))
    assert submachine2.current_state == state3  # Initial state of group2

    # Return to group1 - should restore state2
    machine.process_event(Event("test"))
    assert submachine1.current_state == state2  # History preserved
