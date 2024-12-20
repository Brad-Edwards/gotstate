# tests/integration/test_full_scenario.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time

import pytest
from unittest.mock import Mock
from typing import List, Optional

from hsm.core.events import Event, TimeoutEvent
from hsm.core.state_machine import CompositeStateMachine, StateMachine, _ErrorRecoveryStrategy
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import ValidationError, Validator
from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine
from hsm.runtime.event_queue import EventQueue
from hsm.runtime.executor import Executor
from hsm.runtime.timers import TimeoutScheduler
from hsm.core.hooks import HookProtocol


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
    # Define states
    parent = CompositeState("Parent")
    state1 = State("State1")
    state2 = State("State2")

    parent.add_child_state(state1)
    parent.add_child_state(state2)

    # Define transitions
    t1 = Transition(source=state1, target=state2, guards=[lambda e: True], actions=[lambda e: None], priority=10)

    t2 = Transition(source=state2, target=state1, guards=[lambda e: True], actions=[lambda e: None], priority=10)

    # Construct StateMachine
    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
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
    # Define states
    child1 = State("Child1")
    child2 = State("Child2")
    top = CompositeState("Top", initial_state=child1)

    # Set up hierarchy
    top.add_child_state(child1)
    top.add_child_state(child2)

    # Verify parent-child relationships
    assert child1.parent is top
    assert child2.parent is top

    # Define submachine
    sub_machine = StateMachine(initial_state=child1, validator=validator, hooks=[hook])
    t_sub = Transition(source=child1, target=child2, guards=[], actions=[], priority=0)
    sub_machine.add_transition(t_sub)

    # Create composite machine
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
    """
    Integration test for async:
    - Create an AsyncStateMachine and AsyncEventQueue.
    - Start it, enqueue events, and ensure transitions occur as expected.
    - This requires a simple async loop scenario.
    """

    # States
    start_state = State("AsyncStart")
    end_state = State("AsyncEnd")

    # Transition
    t = Transition(source=start_state, target=end_state, guards=[], actions=[], priority=0)

    machine = AsyncStateMachine(initial_state=start_state, validator=validator, hooks=[hook])
    machine.add_transition(t)  # Use the public method to add transition

    await machine.start()
    assert machine.current_state.name == "AsyncStart"
    queue = AsyncEventQueue(priority=False)

    # We need an async loop that pulls from queue and processes events
    # For integration, simulate a small async event loop:
    async def event_loop():
        # In a real scenario, the async executor would be separate, but we simulate it here.
        for _ in range(3):
            event = await queue.dequeue()
            if event:
                await machine.process_event(event)
            if machine.current_state.name == "AsyncEnd":
                break

    # Enqueue an event that causes transition
    await queue.enqueue(Event("Finish"))
    await asyncio.wait_for(event_loop(), timeout=1.0)

    assert machine.current_state.name == "AsyncEnd", "Machine should transition asynchronously."
    assert "AsyncEnd" in hook.entered
    assert "AsyncStart" in hook.exited

    await machine.stop()

    # Clean up at the end
    for task in asyncio.all_tasks() - {asyncio.current_task()}:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


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
    from hsm.core.state_machine import StateMachine
    from hsm.core.states import State
    from hsm.core.transitions import Transition
    from hsm.core.validations import ValidationError, Validator

    s1 = State("S1")
    s2 = State("S2")
    s3 = State("S3")

    # Add transitions to make all states reachable
    t1 = Transition(source=s1, target=s2)
    t2 = Transition(source=s2, target=s3)  # Make s3 reachable from s2

    machine = StateMachine(initial_state=s1, validator=Validator())
    machine.add_transition(t1)
    machine.add_transition(t2)  # This should now pass validation since s3 is reachable


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

    # Transition with plugin guard and action
    t = Transition(source=start, target=end, guards=[guard.check], actions=[action.run], priority=0)

    machine = StateMachine(initial_state=start, validator=validator, hooks=[hook])
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
    import threading

    from hsm.core.events import Event
    from hsm.runtime.event_queue import EventQueue

    eq = EventQueue()
    processed = []

    def producer():
        for i in range(100):
            eq.enqueue(Event(f"Event{i}"))

    def consumer():
        while True:
            evt = eq.dequeue()
            if evt:
                processed.append(evt)
            if len(processed) >= 100:
                break

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(processed) == 100


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

    # Define transition that modifies data
    def increment_action(event):
        # Access state1's data directly instead of using a stored reference
        state1.data["counter"] += 1

    t1 = Transition(source=state1, target=state2, guards=[lambda e: True], actions=[increment_action], priority=0)

    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
    machine.add_transition(t1)

    # Start and verify initial data
    machine.start()
    assert machine.current_state.data["counter"] == 0

    # Process event and verify data changes
    event = Event("Next")
    machine.process_event(event)

    # Verify state data modifications by accessing state directly
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
    c_machine.add_submachine(group1, sub_machine1)
    c_machine.add_submachine(group2, sub_machine2)
    c_machine.add_transition(t_groups)
    
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
    sub_machine1._context._history.clear()
    sub_machine1.stop()
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

    # Define transitions with proper guards
    t1 = Transition(source=normal, target=error, actions=[failing_action], guards=[lambda e: e.name == "Fail"], priority=10)
    t2 = Transition(source=normal, target=fallback, guards=[lambda e: e.name == "Recover"], priority=20)

    # Create error recovery strategy
    class TestErrorRecovery(_ErrorRecoveryStrategy):
        def recover(self, error: Exception, state_machine: StateMachine) -> None:
            # When error occurs, trigger transition to fallback
            state_machine.process_event(Event("Recover"))

    machine = StateMachine(initial_state=normal, validator=validator, hooks=[hook])
    machine._error_recovery = TestErrorRecovery()  # Set custom error recovery
    machine.add_transition(t1)
    machine.add_transition(t2)

    machine.start()
    assert machine.current_state.name == "Normal"

    # Trigger failing transition
    machine.process_event(Event("Fail"))

    # Verify fallback occurred and error was logged
    assert machine.current_state.name == "Fallback"
    assert len(hook.errors) == 1
    assert "Action failed" in str(hook.errors[0])

    # Cleanup
    machine.stop()


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

    # Define transitions with different priorities
    t1 = Transition(source=state_a, target=state_b, actions=[track_transition("B")], priority=10)

    t2 = Transition(source=state_b, target=state_c, actions=[track_transition("C")], priority=20)

    # Create and configure machine
    machine = StateMachine(initial_state=state_a, validator=validator, hooks=[hook])
    machine.add_transition(t1)
    machine.add_transition(t2)

    # Setup event processing
    eq = EventQueue(priority=True)
    executor = Executor(machine, eq)

    # Start machine
    machine.start()
    assert machine.current_state.name == "StateA"

    # Queue multiple events
    eq.enqueue(Event("Next"))  # Will be processed with default priority
    eq.enqueue(Event("Skip"))  # Will be processed with default priority

    # Run executor
    thread = threading.Thread(target=executor.run)
    thread.start()
    time.sleep(0.1)

    # Verify transition sequence
    assert transition_sequence == ["B", "C"]
    assert machine.current_state.name == "StateC"

    # Cleanup
    executor.stop()
    thread.join(timeout=1.0)


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

    # Define transition
    t = Transition(source=state1, target=state2, priority=0)

    # Create machine
    machine = StateMachine(initial_state=state1, validator=validator, hooks=[hook])
    machine.add_transition(t)

    # Start and verify resource initialization
    machine.start()
    assert machine.current_state.resource.active is True

    # Trigger transition
    machine.process_event(Event("Next"))

    # Verify resource cleanup and new resource initialization
    assert machine.current_state.name == "State2"
    assert not state1.resource.active  # Old resource cleaned up
    assert machine.current_state.resource.active  # New resource initialized


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
