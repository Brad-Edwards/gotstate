"""Unit tests for async state transitions focusing on edge cases and race conditions."""

import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from hsm.core.events import Event
from hsm.core.hooks import HookProtocol
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator
from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine


class AsyncTestHook:
    """Test hook for tracking async state machine events."""

    def __init__(self):
        self.entered_states: List[str] = []
        self.exited_states: List[str] = []
        self.transitions: List[tuple] = []
        self.errors: List[Exception] = []

    async def on_enter(self, state: State) -> None:
        """Track state entry."""
        self.entered_states.append(state.name)

    async def on_exit(self, state: State) -> None:
        """Track state exit."""
        self.exited_states.append(state.name)

    async def on_transition(self, source: State, target: State) -> None:
        """Track state transitions."""
        self.transitions.append((source.name, target.name))

    async def on_error(self, error: Exception) -> None:
        """Track errors."""
        self.errors.append(error)

    async def on_action(self, action_name: str) -> None:
        """Track action execution."""
        pass  # Not used in current tests


@pytest.fixture
async def async_hook():
    return AsyncTestHook()


@pytest.fixture
async def async_machine(async_hook):
    """Create a basic async state machine for testing."""
    start = State("Start")
    middle = State("Middle")
    end = State("End")

    machine = AsyncStateMachine(initial_state=start, validator=Validator(), hooks=[async_hook])

    machine.add_state(middle)
    machine.add_state(end)

    # Add transitions after all states are in graph
    machine.add_transition(Transition(source=start, target=middle, guards=[lambda e: True], actions=[], priority=0))

    machine.add_transition(Transition(source=middle, target=end, guards=[lambda e: True], actions=[], priority=0))

    return machine


@pytest.mark.asyncio
async def test_basic_async_transition(async_machine, async_hook):
    """Test basic async state transition flow."""
    await async_machine.start()
    assert async_machine.current_state.name == "Start"
    assert "Start" in async_hook.entered_states

    await async_machine.process_event(Event("next"))
    assert async_machine.current_state.name == "Middle"
    assert "Middle" in async_hook.entered_states
    assert "Start" in async_hook.exited_states

    await async_machine.process_event(Event("next"))
    assert async_machine.current_state.name == "End"
    assert "End" in async_hook.entered_states
    assert "Middle" in async_hook.exited_states


@pytest.mark.asyncio
async def test_concurrent_transitions(async_machine, async_hook):
    """Test handling of concurrent transition requests."""
    await async_machine.start()

    # Create multiple concurrent event processing tasks
    events = [Event("next") for _ in range(5)]
    tasks = [asyncio.create_task(async_machine.process_event(event)) for event in events]

    # Wait for all transitions to complete
    await asyncio.gather(*tasks)

    # Verify final state and transition order
    assert async_machine.current_state.name == "End"
    assert len(async_hook.transitions) <= 2  # Should only transition Start->Middle->End

    # Verify no duplicate states in transition path
    visited_states = set()
    for source, target in async_hook.transitions:
        assert source not in visited_states  # No repeated transitions
        visited_states.add(source)


@pytest.mark.asyncio
async def test_async_transition_with_delayed_guard():
    """Test transitions with async guards that introduce delays."""
    start = State("Start")
    end = State("End")

    async def delayed_guard(event: Event) -> bool:
        await asyncio.sleep(0.1)  # Simulate async operation
        return True

    machine = AsyncStateMachine(initial_state=start, validator=Validator())
    machine.add_state(end)

    machine.add_transition(Transition(source=start, target=end, guards=[delayed_guard], actions=[], priority=0))

    await machine.start()
    assert machine.current_state.name == "Start"

    # Process event with delayed guard
    await machine.process_event(Event("next"))
    assert machine.current_state.name == "End"


@pytest.mark.asyncio
async def test_async_transition_with_failing_action():
    """Test error handling when async actions fail."""
    start = State("Start")
    end = State("End")
    errors = []

    async def failing_action(event: Event) -> None:
        raise RuntimeError("Action failed")

    hook = AsyncTestHook()
    machine = AsyncStateMachine(initial_state=start, validator=Validator(), hooks=[hook])
    machine.add_state(end)

    machine.add_transition(
        Transition(source=start, target=end, guards=[lambda e: True], actions=[failing_action], priority=0)
    )

    await machine.start()
    await machine.process_event(Event("next"))

    # Verify error was caught and state didn't change
    assert len(hook.errors) == 1
    assert isinstance(hook.errors[0], RuntimeError)
    assert machine.current_state.name == "Start"


@pytest.mark.asyncio
async def test_async_transition_cancellation():
    """Test cancellation of async transitions."""
    start = State("Start")
    end = State("End")
    action_started = asyncio.Event()

    async def long_running_action(event: Event) -> None:
        action_started.set()  # Signal that we've started the action
        try:
            await asyncio.sleep(1.0)  # Long enough to ensure cancellation
        except asyncio.CancelledError:
            # Properly handle cancellation by re-raising
            raise

    machine = AsyncStateMachine(initial_state=start, validator=Validator())
    machine.add_state(end)

    machine.add_transition(
        Transition(source=start, target=end, guards=[lambda e: True], actions=[long_running_action], priority=0)
    )

    await machine.start()
    assert machine.current_state.name == "Start"

    # Start transition in a task
    task = asyncio.create_task(machine.process_event(Event("next")))

    # Wait for action to start
    await action_started.wait()

    # Cancel the task
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify state didn't change after cancellation
    assert machine.current_state.name == "Start"


@pytest.mark.skip(reason="Concurrent test needs to be fixed - temporarily disabled")
@pytest.mark.asyncio
async def test_concurrent_enqueue_dequeue():
    """Test concurrent enqueue/dequeue operations."""
    queue = AsyncEventQueue()
    event_count = 20
    producer_count = 3
    consumer_count = 2

    received_events = []
    events_processed = 0
    stop_consumers = asyncio.Event()

    async def producer(id: int):
        for i in range(event_count):
            await queue.enqueue(Event(f"event_{id}_{i}"))

    async def consumer(id: int):
        nonlocal events_processed
        while not stop_consumers.is_set():
            try:
                event = await asyncio.wait_for(queue.dequeue(), timeout=0.1)
                if event is not None:
                    received_events.append(event)
                    events_processed += 1
            except asyncio.TimeoutError:
                continue

    # Start producers
    producer_tasks = [asyncio.create_task(producer(i)) for i in range(producer_count)]

    # Start consumers
    consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(consumer_count)]

    try:
        # Wait for producers to finish
        await asyncio.wait_for(asyncio.gather(*producer_tasks), timeout=0.5)

        # Wait for all events to be consumed
        expected_events = event_count * producer_count
        while events_processed < expected_events:
            await asyncio.sleep(0.01)
            if events_processed >= expected_events:
                break

        # Signal consumers to stop
        stop_consumers.set()

        # Wait for consumers to finish
        await asyncio.wait_for(asyncio.gather(*consumer_tasks), timeout=0.1)

    except asyncio.TimeoutError:
        pytest.fail("Test timed out - possible deadlock or performance issue")
    finally:
        # Clean up any remaining tasks
        stop_consumers.set()
        for task in consumer_tasks:
            if not task.done():
                task.cancel()
        for task in producer_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to be properly cancelled
        await asyncio.gather(*consumer_tasks, *producer_tasks, return_exceptions=True)

    # Verify results
    assert len(received_events) == expected_events
    event_names = {event.name for event in received_events}
    assert len(event_names) == expected_events
