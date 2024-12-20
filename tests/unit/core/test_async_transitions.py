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
        self.entered_states.append(state.name)

    async def on_exit(self, state: State) -> None:
        self.exited_states.append(state.name)

    async def on_transition(self, source: State, target: State) -> None:
        self.transitions.append((source.name, target.name))

    async def on_error(self, error: Exception) -> None:
        self.errors.append(error)


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

    async def long_running_action(event: Event) -> None:
        await asyncio.sleep(1.0)  # Long running action

    machine = AsyncStateMachine(initial_state=start, validator=Validator())
    machine.add_state(end)

    machine.add_transition(
        Transition(source=start, target=end, guards=[lambda e: True], actions=[long_running_action], priority=0)
    )

    await machine.start()

    # Start transition but cancel it immediately
    task = asyncio.create_task(machine.process_event(Event("next")))
    await asyncio.sleep(0.1)  # Give time for transition to start
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify state didn't change after cancellation
    assert machine.current_state.name == "Start"
