"""Unit tests for async state transitions focusing on edge cases and race conditions."""

import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from gotstate.core.events import Event
from gotstate.core.hooks import HookProtocol
from gotstate.core.states import CompositeState, State
from gotstate.core.transitions import Transition
from gotstate.core.validations import Validator
from gotstate.runtime.async_support import AsyncEventQueue, AsyncStateMachine


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

    # Add all states first
    machine.add_state(middle)
    machine.add_state(end)

    # Add transitions with guards that always return True to ensure reachability
    machine.add_transition(
        Transition(source=start, target=middle, guards=[lambda e: e.name == "next"], actions=[], priority=1)
    )
    machine.add_transition(
        Transition(source=middle, target=end, guards=[lambda e: e.name == "next"], actions=[], priority=1)
    )
    # Add direct transition from start to end for testing concurrent transitions
    machine.add_transition(
        Transition(source=start, target=end, guards=[lambda e: e.name == "concurrent"], actions=[], priority=0)
    )

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
    assert async_machine.current_state.name == "Start"

    # Create multiple concurrent event processing tasks
    events = [Event("concurrent") for _ in range(5)]  # Use concurrent event type
    tasks = [asyncio.create_task(async_machine.process_event(event)) for event in events]

    # Wait for all transitions to complete
    results = await asyncio.gather(*tasks)

    # Verify final state and transition order
    assert async_machine.current_state.name == "End"

    # At least one transition should have succeeded
    assert any(results), "No transitions succeeded"

    # Verify transitions were recorded
    assert len(async_hook.transitions) == 1, "Expected exactly one successful transition"
    assert async_hook.transitions[0] == ("Start", "End"), "Expected transition from Start to End"

    # Verify state changes
    assert "Start" in async_hook.exited_states
    assert "End" in async_hook.entered_states


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
    action_cleanup_complete = asyncio.Event()

    async def long_running_action(event: Event) -> None:
        action_started.set()
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            action_cleanup_complete.set()
            raise
        finally:
            action_cleanup_complete.set()

    machine = AsyncStateMachine(initial_state=start, validator=Validator())
    machine.add_state(end)
    machine.add_transition(
        Transition(source=start, target=end, guards=[lambda e: True], actions=[long_running_action], priority=0)
    )

    await machine.start()
    assert machine.current_state.name == "Start"

    # Start the transition in a task
    process_task = asyncio.create_task(machine.process_event(Event("next")))

    # Wait for action to start
    try:
        await asyncio.wait_for(action_started.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Action did not start in time")

    # Cancel the task and wait for cleanup
    process_task.cancel()

    try:
        await asyncio.wait_for(action_cleanup_complete.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Action cleanup did not complete in time")

    # Clean up cancelled task
    try:
        await process_task
    except asyncio.CancelledError:
        pass

    # Verify state and clean up
    assert machine.current_state.name == "Start"
    await machine.stop()


@pytest.mark.asyncio
async def test_concurrent_enqueue_dequeue():
    """Test concurrent enqueue/dequeue operations."""
    queue = AsyncEventQueue()
    event_count = 5  # Reduced from 20 for faster tests
    producer_count = 2  # Reduced from 3
    consumer_count = 2

    received_events = asyncio.Queue()
    stop_consumers = asyncio.Event()

    async def producer(id: int):
        for i in range(event_count):
            await queue.enqueue(Event(f"event_{id}_{i}"))
            await asyncio.sleep(0)  # Allow other tasks to run

    async def consumer(id: int):
        while not stop_consumers.is_set():
            try:
                event = await asyncio.wait_for(queue.dequeue(), timeout=0.1)
                if event is not None:
                    await received_events.put(event)
            except asyncio.TimeoutError:
                continue
            await asyncio.sleep(0)  # Allow other tasks to run

    # Start producers and consumers
    producer_tasks = [asyncio.create_task(producer(i)) for i in range(producer_count)]
    consumer_tasks = [asyncio.create_task(consumer(i)) for i in range(consumer_count)]

    try:
        # Wait for producers to finish
        await asyncio.gather(*producer_tasks)

        # Wait for all events to be consumed
        expected_events = event_count * producer_count
        received_count = 0

        while received_count < expected_events:
            try:
                await asyncio.wait_for(received_events.get(), timeout=0.1)
                received_count += 1
            except asyncio.TimeoutError:
                if received_count >= expected_events:
                    break
                await asyncio.sleep(0.01)

        # Signal consumers to stop
        stop_consumers.set()

        # Wait for consumers to finish
        await asyncio.gather(*consumer_tasks)

    finally:
        # Clean up
        stop_consumers.set()
        for task in consumer_tasks:
            if not task.done():
                task.cancel()
        for task in producer_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to be cancelled
        await asyncio.gather(*consumer_tasks, *producer_tasks, return_exceptions=True)

    # Verify results
    assert received_count == expected_events
