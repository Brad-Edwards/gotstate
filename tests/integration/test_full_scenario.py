# tests/integration/test_full_scenario.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time

import pytest

from hsm.core.events import Event, TimeoutEvent
from hsm.core.state_machine import CompositeStateMachine, StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator
from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine
from hsm.runtime.event_queue import EventQueue
from hsm.runtime.executor import Executor
from hsm.runtime.timers import TimeoutScheduler


@pytest.fixture
def hook():
    """A hook to capture lifecycle events for assertions."""

    class TestHook:
        def __init__(self):
            self.entered = []
            self.exited = []
            self.errors = []

        def on_enter(self, state):
            self.entered.append(state.name)

        def on_exit(self, state):
            self.exited.append(state.name)

        def on_error(self, error):
            self.errors.append(str(error))

    return TestHook()


@pytest.fixture
def validator():
    """A validator that ensures states and transitions have names and guards are callable."""
    # The default Validator might be no-op. We can still rely on it not failing a correct setup.
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
    top = CompositeState("Top")
    child1 = State("Child1")
    child2 = State("Child2")

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
    s3 = State("S3")  # This will be unreachable
    # Add a transition from s1 to s2
    t1 = Transition(source=s1, target=s2)
    # Add a transition from s3 to s2 - s3 is unreachable because no transition leads to it
    t2 = Transition(source=s3, target=s2)

    machine = StateMachine(initial_state=s1, validator=Validator())
    machine.add_transition(t1)
    # This should fail validation because s3 will be added to states but is unreachable
    with pytest.raises(ValidationError):
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
