# tests/integration/test_full_scenario.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time

import pytest

from hsm.core.actions import BasicActions
from hsm.core.events import Event, TimeoutEvent
from hsm.core.guards import BasicGuards
from hsm.core.hooks import HookManager
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
    Full integration test:
    - Create states with entry/exit actions.
    - Add transitions with guards and actions.
    - Use hooks to track state machine lifecycle.
    - Run via Executor and EventQueue.
    - Ensure final states and hooks are as expected.
    """

    # Define actions
    def entry_action_fn(**kwargs):
        # Imagine logging or setting data
        pass

    def exit_action_fn(**kwargs):
        pass

    def transition_action_fn(**kwargs):
        pass

    entry_actions = [lambda: BasicActions.execute(entry_action_fn)]
    exit_actions = [lambda: BasicActions.execute(exit_action_fn)]
    transition_actions = [lambda: BasicActions.execute(transition_action_fn)]

    # Define states
    idle_state = State("Idle", entry_actions=entry_actions, exit_actions=exit_actions)
    working_state = State("Working", entry_actions=entry_actions, exit_actions=exit_actions)

    # Define transitions
    # Guard checks a simple condition function returning True
    def guard_condition():
        return True

    guards = [lambda event: BasicGuards.check_condition(guard_condition)]

    t = Transition(source=idle_state, target=working_state, guards=guards, actions=transition_actions, priority=10)

    # Construct StateMachine
    machine = StateMachine(initial_state=idle_state, validator=validator, hooks=[hook])
    # We assume a method to add transitions is available or transitions are configured in constructor.
    # If not, consider a machine.add_transition() method. Since not explicitly defined, we rely on
    # hypothetical public methods or the machine context. For integration testing, we assume availability:
    machine._StateMachine__context.add_transition(t)  # Accessing internal context due to prior design assumption.

    # Validate machine
    machine.validator.validate_state_machine(machine)

    # Setup runtime
    eq = EventQueue(priority=False)
    executor = Executor(machine, eq)

    # Start machine
    machine.start()
    assert machine.current_state.name == "Idle"
    assert "Idle" in hook.entered

    # Process a normal event causing transition to "Working"
    eq.enqueue(Event("StartWork"))
    # Run executor in a separate thread to emulate real runtime
    stop_flag = threading.Event()

    def run_executor():
        executor.run()
        stop_flag.set()

    thread = threading.Thread(target=run_executor)
    thread.start()

    # Give the executor time to process
    time.sleep(0.1)
    # At this point, machine should have transitioned
    assert machine.current_state.name == "Working"
    assert "Working" in hook.entered
    assert "Idle" in hook.exited

    # Stop machine and executor
    machine.stop()
    executor.stop()
    thread.join(timeout=2.0)
    assert stop_flag.is_set()

    # No errors should have occurred
    assert len(hook.errors) == 0


def test_composite_state_machine_integration(hook, validator):
    """
    Integration test for composite state machines:
    - Create a composite state with a nested submachine.
    - Validate transitions occur in the nested structure.
    """

    # Define states
    top = CompositeState("Top")
    child1 = State("Child1")
    child2 = State("Child2")
    top.add_child_state(child1)
    top.add_child_state(child2)

    # Transition within top-level machine
    t_top = Transition(source=top, target=top, guards=[], actions=[], priority=0)

    # Define a submachine for top
    sub_machine = StateMachine(initial_state=child1, validator=validator, hooks=[hook])
    t_sub = Transition(source=child1, target=child2, guards=[], actions=[], priority=0)
    sub_machine._StateMachine__context.add_transition(t_sub)

    # Composite machine manages top and sub_machine
    c_machine = CompositeStateMachine(initial_state=top, validator=validator, hooks=[hook])
    c_machine._StateMachine__context.add_transition(t_top)
    c_machine.add_submachine(top, sub_machine)

    # Validate and start
    c_machine.validator.validate_state_machine(c_machine)
    c_machine.start()
    assert c_machine.current_state.name == "Top"
    # submachine should not start automatically unless designed so; let's start it:
    sub_machine.start()
    assert sub_machine.current_state.name == "Child1"

    # Trigger a submachine transition
    sub_machine.process_event(Event("Go"))
    assert sub_machine.current_state.name == "Child2"
    # Hooks should have recorded entering Child2, exiting Child1
    assert "Child2" in hook.entered
    assert "Child1" in hook.exited

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
    machine._AsyncStateMachine__context = (
        machine._AsyncStateMachine__context
    )  # If such a context is parallel to sync machine
    machine._AsyncStateMachine__context.add_transition(
        t
    )  # Hypothetical internal method, or a public API if implemented.

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
    # Create an invalid machine: no initial state or transitions referencing unknown states
    from hsm.core.state_machine import StateMachine
    from hsm.core.states import State
    from hsm.core.transitions import Transition
    from hsm.core.validations import ValidationError, Validator

    s1 = State("S1")
    s2 = State("S2")
    # Transition referencing s2 but never added to machine properly or no route to s2?
    t = Transition(source=s1, target=s2)
    machine = StateMachine(initial_state=s1, validator=Validator())

    # Add transition incorrectly, let's say we just do this:
    machine._StateMachine__context.add_transition(t)

    # Attempt validation
    with pytest.raises(ValidationError):
        machine.validator.validate_state_machine(machine)


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
    machine._StateMachine__context.add_transition(t)

    eq = EventQueue(priority=False)
    executor = Executor(machine, eq)
    machine.start()

    eq.enqueue(Event("Go"))

    stop_flag = threading.Event()

    def run_exec():
        executor.run()
        stop_flag.set()

    thread = threading.Thread(target=run_exec)
    thread.start()

    time.sleep(0.1)
    assert machine.current_state.name == "PluginEnd"
    machine.stop()
    executor.stop()
    thread.join()

    # Check hooks
    assert "PluginEnd" in hook.entered
    assert "PluginStart" in hook.exited
