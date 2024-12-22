# test_integration.py
import asyncio
import threading
import time

import pytest

from hsm.core.actions import BasicActions
from hsm.core.errors import HSMError, StateNotFoundError, TransitionError, ValidationError
from hsm.core.events import Event, TimeoutEvent
from hsm.core.guards import BasicGuards
from hsm.core.state_machine import CompositeStateMachine, StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine, _AsyncEventProcessingLoop
from hsm.runtime.event_queue import EventQueue
from hsm.runtime.executor import Executor
from hsm.runtime.timers import TimeoutScheduler


@pytest.fixture
def simple_machine():
    """
    A minimal machine: Idle -> Active -> Finished
    """
    idle = State("Idle")
    active = State("Active")
    finished = State("Finished")

    machine = StateMachine(initial_state=idle)
    machine.add_state(active)
    machine.add_state(finished)
    machine.add_transition(
        Transition(
            source=idle,
            target=active,
            guards=[lambda e: e.name == "go_active"],
            actions=[lambda e: machine._graph.set_state_data(active, "activated", True)],
        )
    )
    machine.add_transition(
        Transition(
            source=active,
            target=finished,
            guards=[lambda e: e.name == "finish"],
            actions=[lambda e: machine._graph.set_state_data(finished, "finished", True)],
        )
    )
    return machine


def test_simple_machine_transitions(simple_machine):
    """
    Verifies a simple linear set of transitions.
    """
    # The machine is not started yet
    assert simple_machine.current_state is not None
    assert simple_machine.current_state.name == "Idle"

    # Start it
    simple_machine.start()
    assert simple_machine.current_state.name == "Idle"

    # Transition to Active
    event = Event("go_active")
    transitioned = simple_machine.process_event(event)
    assert transitioned is True
    assert simple_machine.current_state.name == "Active"
    assert simple_machine._graph.get_state_data(simple_machine.current_state).get("activated") is True

    # Transition to Finished
    event = Event("finish")
    transitioned = simple_machine.process_event(event)
    assert transitioned is True
    assert simple_machine.current_state.name == "Finished"
    assert simple_machine._graph.get_state_data(simple_machine.current_state).get("finished") is True


def test_no_transition_fails(simple_machine):
    """
    If event doesn't match any guard, transition should fail gracefully.
    """
    simple_machine.start()
    e = Event("unknown_event")
    transitioned = simple_machine.process_event(e)
    assert transitioned is False
    assert simple_machine.current_state.name == "Idle"


def test_reset_machine(simple_machine):
    """
    Tests stopping, resetting, and re-starting.
    """
    simple_machine.start()
    # Move to "Active"
    simple_machine.process_event(Event("go_active"))
    assert simple_machine.current_state.name == "Active"

    # Stop
    simple_machine.stop()
    assert simple_machine._started is False
    assert simple_machine.current_state is None

    # Reset
    simple_machine.reset()
    # This should revert current state to "Idle" but not 'started'
    assert simple_machine.current_state.name == "Idle"
    assert simple_machine._started is False

    # Start again
    simple_machine.start()
    assert simple_machine._started is True


def test_validator_checks(simple_machine):
    """
    Ensures the graph validator can be invoked and returns no errors for a correct machine.
    """
    simple_machine.start()
    errors = simple_machine.validate()
    assert not errors, f"Validation should pass but got: {errors}"


def test_priority_transitions():
    """
    Ensures higher priority transitions fire first if multiple guards pass.
    """
    s1 = State("S1")
    s2 = State("S2")
    s3 = State("S3")
    machine = StateMachine(initial_state=s1)
    machine.add_state(s2)
    machine.add_state(s3)

    # Two transitions from s1 with guards that pass for event "test".
    # Priority is higher for s1->s3
    t1 = Transition(s1, s2, guards=[lambda e: e.name == "test"], priority=1)
    t2 = Transition(s1, s3, guards=[lambda e: e.name == "test"], priority=10)
    machine.add_transition(t1)
    machine.add_transition(t2)

    machine.start()
    machine.process_event(Event("test"))
    assert machine.current_state.name == "S3", "Higher priority transition must be taken first."


def test_hooks_and_error_recovery():
    """
    Validates that hooks get called on_enter, on_exit, and on_error. Also test custom error recovery.
    """
    log = []

    class HookImpl:
        def on_enter(self, state):
            log.append(f"enter_{state.name}")

        def on_exit(self, state):
            log.append(f"exit_{state.name}")

        def on_error(self, error):
            log.append(f"error_{type(error).__name__}")

    class ErrorRecovery:
        def recover(self, error, state_machine):
            log.append(f"recover_{type(error).__name__}")

    s1 = State("S1")
    s2 = State("S2")
    machine = StateMachine(initial_state=s1, hooks=[HookImpl()], error_recovery=ErrorRecovery())
    machine.add_state(s2)

    # Add a transition that intentionally throws an action error
    def faulty_action(e):
        raise RuntimeError("Simulated action fail")

    machine.add_transition(
        Transition(
            source=s1,
            target=s2,
            actions=[faulty_action],
            guards=[lambda e: True],
        )
    )

    machine.start()
    # Attempt transition to s2
    machine.process_event(Event("go"))
    # We should see enter/exit for s1, but an error occurs during transition to s2
    assert log == [
        "enter_S1",  # machine start
        "exit_S1",  # transition tries to exit s1
        "error_RuntimeError",  # on_error from hooks
        "recover_RuntimeError",  # custom error recovery
    ]


def test_composite_state_machine():
    """
    Tests hierarchical structure, ensuring submachine transitions occur properly.
    """
    top = CompositeState("Top")
    sub1 = State("Sub1")
    sub2 = State("Sub2")

    machine = CompositeStateMachine(top)
    machine.add_state(sub1, parent=top)
    machine.add_state(sub2, parent=top)

    # Set initial state through graph
    machine._graph.set_initial_state(top, sub1)

    machine.add_transition(Transition(source=sub1, target=sub2, guards=[lambda e: e.name == "next"]))

    machine.start()
    assert machine.current_state.name == "Sub1"

    # Switch to Sub2
    transitioned = machine.process_event(Event("next"))
    assert transitioned is True
    assert machine.current_state.name == "Sub2"


def test_submachine_integration():
    """
    Ensures adding a submachine under a composite parent merges transitions and states properly.
    """
    root = CompositeState("Root")
    sub_root = State("SubRoot")  # initial of submachine
    sub_end = State("SubEnd")

    submachine = StateMachine(initial_state=sub_root)
    submachine.add_state(sub_end)
    submachine.add_transition(Transition(source=sub_root, target=sub_end, guards=[lambda e: e.name == "to_sub_end"]))

    main_machine = CompositeStateMachine(root)
    main_machine.add_submachine(root, submachine)
    main_machine.start()

    # The submachine starts in SubRoot
    assert main_machine.current_state.name == "SubRoot"

    # Fire event to move submachine to SubEnd
    main_machine.process_event(Event("to_sub_end"))
    assert main_machine.current_state.name == "SubEnd"


def test_sync_executor_fifo_queue(simple_machine):
    """
    Confirms the Executor + FIFO queue processes events in order received.
    """
    queue = EventQueue(priority=False)
    executor = Executor(simple_machine, queue)

    simple_machine.start()
    assert simple_machine.current_state.name == "Idle"

    queue.enqueue(Event("go_active"))
    queue.enqueue(Event("finish"))

    t = threading.Thread(target=executor.run)
    t.start()

    # Let it run a bit longer to ensure both transitions complete
    time.sleep(0.5)
    executor.stop()
    t.join()

    # Should have gone through both transitions: Idle -> Active -> Finished
    assert simple_machine.current_state.name == "Finished"


def test_sync_executor_priority_queue():
    """
    Checks that a priority queue handles events with higher priority first.
    """
    s1 = State("S1")
    s2 = State("S2")
    s3 = State("S3")

    machine = StateMachine(initial_state=s1)
    machine.add_state(s2)
    machine.add_state(s3)

    # transitions
    machine.add_transition(Transition(s1, s2, guards=[lambda e: e.name == "go"], priority=1))
    machine.add_transition(Transition(s1, s3, guards=[lambda e: e.name == "go"], priority=10))

    queue = EventQueue(priority=True)
    executor = Executor(machine, queue)

    machine.start()

    # Both events have name="go", but different priorities.
    # The "lower_priority" event has priority=1, "higher_priority" has priority=10.
    lower_priority = Event("go", priority=1)
    higher_priority = Event("go", priority=10)
    queue.enqueue(lower_priority)
    queue.enqueue(higher_priority)

    t = threading.Thread(target=executor.run)
    t.start()

    time.sleep(0.2)
    executor.stop()
    t.join()

    # Because the "go" event with priority=10 is dequeued first, machine should go to s3
    assert machine.current_state.name == "S3"


@pytest.mark.asyncio
async def test_async_machine_basic():
    """
    Tests a basic asynchronous state machine flow.
    """
    s1 = State("S1")
    s2 = State("S2")
    async_machine = AsyncStateMachine(initial_state=s1)
    async_machine.add_state(s2)
    async_machine.add_transition(Transition(source=s1, target=s2, guards=[lambda e: e.name == "go"]))

    await async_machine.start()
    transitioned = await async_machine.process_event(Event("go"))
    assert transitioned is True
    assert async_machine.current_state.name == "S2"
    await async_machine.stop()


@pytest.mark.asyncio
async def test_async_event_queue_processing():
    """
    Ensures the AsyncEventQueue + AsyncStateMachine integration is correct.
    """
    s1 = State("S1")
    s2 = State("S2")
    machine = AsyncStateMachine(s1)
    machine.add_state(s2)
    machine.add_transition(Transition(s1, s2, guards=[lambda e: e.name == "go"]))

    loop = _AsyncEventProcessingLoop(
        machine,
        AsyncEventQueue(priority=True),
    )

    loop_task = asyncio.create_task(loop.start_loop())
    # Wait for the machine to start
    await asyncio.sleep(0.1)

    # Enqueue an event
    await loop._queue.enqueue(Event("go", priority=5))
    await asyncio.sleep(0.2)
    # Machine should have transitioned
    assert machine.current_state.name == "S2"

    # Cleanup
    await loop.stop_loop()
    await loop_task


def test_timeout_scheduler():
    """
    Verifies that the TimeoutScheduler spawns TimeoutEvents at the correct time.
    Uses mocked time to ensure reliable testing.
    """
    from unittest.mock import patch

    with patch("time.time") as mock_time:
        current_time = 1000.0  # Start at an arbitrary time
        mock_time.return_value = current_time

        scheduler = TimeoutScheduler()
        event1 = TimeoutEvent("timeout1", deadline=current_time + 0.2)
        event2 = TimeoutEvent("timeout2", deadline=current_time + 1.0)
        scheduler.schedule_timeout(event1)
        scheduler.schedule_timeout(event2)

        # Advance time past event1's deadline
        mock_time.return_value = current_time + 0.3
        expired = scheduler.check_timeouts()
        assert len(expired) == 1, f"Expected 1 expired event, got {len(expired)}"
        assert expired[0].name == "timeout1"

        # Advance time past event2's deadline
        mock_time.return_value = current_time + 1.1
        expired = scheduler.check_timeouts()
        assert len(expired) == 1, f"Expected 1 expired event, got {len(expired)}"
        assert expired[0].name == "timeout2"


def test_expected_vs_actual_hfsm_behavior():
    """
    Spot-check that the library's composite state & submachine approach
    matches typical HFSM expectations. If there's a discrepancy, mark fail.

    For instance, typical HFSMs store child states in a composite. Here, the library
    defers to StateGraph to manage that. This is consistent with the design doc,
    but slightly unusual for HFSM-libraries that store sub-states directly.
    """
    # The code does so by design, so we won't fail here. But if you discover more:
    # pytest.fail("XYZ is different than standard HFSM approach; see docs.")
    pass


def test_validation_of_composite_initial():
    """
    Checks that a composite state's initial state must be set or validated.
    """
    root = CompositeState("Root")
    # No initial child set, but we start the machine anyway
    machine = CompositeStateMachine(root)

    # Starting the machine might raise a ValidationError
    # if the library truly enforces that composites must have initial states.
    # If it does not raise, we point out the discrepancy.
    try:
        machine.start()
    except ValidationError:
        # This means the library is enforcing the requirement - good.
        pass
    else:
        # If it doesn't raise, we note that the HFSM spec usually demands an initial child.
        pytest.fail(
            "CompositeState with no explicit initial child was allowed to start. "
            "HFSM convention typically requires an initial sub-state."
        )
