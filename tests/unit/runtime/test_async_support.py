# tests/unit/test_async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hsm.core.events import Event
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import AsyncValidator
from hsm.runtime.async_support import (
    AsyncCompositeStateMachine,
    AsyncEventQueue,
    AsyncStateMachine,
    _AsyncEventProcessingLoop,
    _AsyncLock,
)


@pytest.mark.asyncio
async def test_async_state_machine_init(dummy_state, validator):
    """Test initialization of AsyncStateMachine."""
    asm = AsyncStateMachine(initial_state=dummy_state, validator=validator)
    assert asm._initial_state == dummy_state
    assert asm.current_state is None


@pytest.mark.asyncio
async def test_async_state_machine_start_stop(dummy_state):
    """Test starting and stopping AsyncStateMachine."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    dummy_state.on_enter = MagicMock()
    await asm.start()
    dummy_state.on_enter.assert_called_once()
    await asm.stop()
    dummy_state.on_exit = MagicMock()


@pytest.mark.asyncio
async def test_async_state_machine_process_event(dummy_state, mock_event):
    """Test event processing in AsyncStateMachine."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    await asm.start()
    # With no transitions, process_event should not fail
    await asm.process_event(mock_event)


@pytest.mark.asyncio
async def test_async_event_queue(mock_event):
    """Test basic AsyncEventQueue operations."""
    eq = AsyncEventQueue(priority=False)
    await eq.enqueue(mock_event)
    out = await eq.dequeue()
    assert out == mock_event
    await eq.clear()
    empty = await eq.dequeue()
    assert empty is None
    assert eq.priority_mode is False


@pytest.mark.asyncio
async def test_async_lock():
    """Test AsyncLock functionality."""
    lock = _AsyncLock()
    await lock.acquire()
    # Verify lock is acquired
    lock.release()
    # Verify lock can be reacquired
    await lock.acquire()
    lock.release()


@pytest.mark.asyncio
async def test_async_state_machine_double_start(dummy_state):
    """Test starting AsyncStateMachine multiple times."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    dummy_state.on_enter = MagicMock()
    await asm.start()
    await asm.start()  # Second start should be ignored
    dummy_state.on_enter.assert_called_once()


@pytest.mark.asyncio
async def test_async_state_machine_transition(dummy_state, mock_event):
    """Test state transitions in AsyncStateMachine."""
    target_state = State("target")
    transition = Transition(source=dummy_state, target=target_state, guards=[lambda e: True], actions=[MagicMock()])

    asm = AsyncStateMachine(initial_state=dummy_state)
    asm.add_state(target_state)
    asm.add_transition(transition)
    await asm.start()
    await asm.process_event(mock_event)

    assert asm.current_state == target_state
    transition.actions[0].assert_called_once_with(mock_event)


@pytest.mark.asyncio
async def test_async_composite_state_machine():
    """Test AsyncCompositeStateMachine functionality."""
    root = CompositeState("root")
    sub1 = State("sub1")
    sub2 = State("sub2")

    # Create main machine
    main_machine = AsyncCompositeStateMachine(initial_state=root)

    # Create submachine
    sub_machine = AsyncStateMachine(initial_state=sub1)
    sub_machine.add_state(sub2)

    # Add transition after all states are set up
    sub_machine.add_transition(Transition(source=sub1, target=sub2, guards=[lambda e: e.name == "next"]))

    # Add submachine to main machine
    main_machine.add_submachine(root, sub_machine)

    # Start and verify initial state
    await main_machine.start()
    assert main_machine.current_state == sub1

    # Test transition within submachine
    await main_machine.process_event(Event("next"))
    assert main_machine.current_state == sub2


@pytest.mark.asyncio
async def test_async_event_processing_loop():
    """Test AsyncEventProcessingLoop functionality."""
    initial_state = State("initial")
    machine = AsyncStateMachine(initial_state=initial_state)
    queue = AsyncEventQueue()

    loop = _AsyncEventProcessingLoop(machine, queue)

    # Start loop in background task
    task = asyncio.create_task(loop.start_loop())

    # Give loop time to start
    await asyncio.sleep(0.1)

    # Stop loop
    await loop.stop_loop()

    # Clean up task
    await task


@pytest.mark.asyncio
async def test_async_event_queue_priority_mode():
    """Test priority mode in AsyncEventQueue."""
    eq = AsyncEventQueue(priority=True)
    assert eq.priority_mode is True

    await eq.enqueue(Event("first", priority=1))
    await eq.enqueue(Event("second", priority=1))

    first = await eq.dequeue()
    second = await eq.dequeue()
    assert first.name == "first"
    assert second.name == "second"


@pytest.mark.asyncio
async def test_async_state_machine_error_handling(dummy_state, mock_event):
    """Test error handling in AsyncStateMachine."""
    error_hook = MagicMock()

    async def async_on_enter(state):
        pass

    error_hook.on_enter = async_on_enter
    error_hook.on_error = MagicMock()

    asm = AsyncStateMachine(initial_state=dummy_state, hooks=[error_hook])
    await asm.start()

    # Cause an error by making the state's on_exit throw
    dummy_state.on_exit = MagicMock(side_effect=Exception("Test error"))

    # Add a transition that will trigger the error
    target_state = State("target")
    asm.add_state(target_state)
    transition = Transition(source=dummy_state, target=target_state, guards=[lambda e: True])
    asm.add_transition(transition)

    await asm.process_event(mock_event)
    error_hook.on_error.assert_called_once()


@pytest.mark.asyncio
async def test_async_event_queue_timeout():
    """Test timeout behavior in AsyncEventQueue."""
    eq = AsyncEventQueue()
    # Should return None when queue is empty (after timeout)
    result = await eq.dequeue()
    assert result is None


@pytest.mark.asyncio
async def test_async_state_machine_process_event_when_stopped(dummy_state, mock_event):
    """Test event processing when machine is stopped."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    # Don't start the machine
    await asm.process_event(mock_event)  # Should do nothing
    assert asm.current_state is None


@pytest.mark.asyncio
async def test_async_state_machine_validator_integration():
    """Test async validator integration."""

    class CustomAsyncValidator(AsyncValidator):
        async def validate_state_machine(self, machine):
            # Simulate async validation
            await asyncio.sleep(0.1)
            await super().validate_state_machine(machine)

    start_state = State("Start")
    machine = AsyncStateMachine(initial_state=start_state, validator=CustomAsyncValidator())
    await machine.start()
    await machine.stop()


@pytest.mark.asyncio
async def test_async_state_machine_with_async_guards():
    """Test AsyncStateMachine with async guard functions."""
    initial = State("initial")
    target = State("target")

    async def async_guard(event):
        await asyncio.sleep(0.1)
        return True

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, guards=[async_guard]))

    await machine.start()
    await machine.process_event(Event("test"))
    assert machine.current_state == target


@pytest.mark.asyncio
async def test_async_state_machine_with_async_actions():
    """Test AsyncStateMachine with async actions."""
    initial = State("initial")
    target = State("target")
    action_executed = False

    async def async_action(event):
        nonlocal action_executed
        await asyncio.sleep(0.1)
        action_executed = True

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, guards=[lambda e: True], actions=[async_action]))

    await machine.start()
    await machine.process_event(Event("test"))
    assert action_executed is True


@pytest.mark.asyncio
async def test_async_composite_state_machine_boundary_transitions():
    """Test boundary transitions in AsyncCompositeStateMachine."""
    # Create states
    root = CompositeState("root")
    sub1 = State("sub1")
    sub2 = State("sub2")
    external = State("external")

    # Create submachine
    sub_machine = AsyncStateMachine(initial_state=sub1)
    sub_machine.add_state(sub2)

    # Add transition after states are set up
    sub_machine.add_transition(Transition(source=sub1, target=sub2, guards=[lambda e: e.name == "internal"]))

    # Create main machine
    main_machine = AsyncCompositeStateMachine(initial_state=root)
    main_machine.add_state(external)
    main_machine.add_submachine(root, sub_machine)

    # Add boundary transition
    main_machine.add_transition(
        Transition(source=root, target=external, guards=[lambda e: e.name == "exit_submachine"])
    )

    # Start and verify initial state
    await main_machine.start()
    assert main_machine.current_state == sub1

    # Test internal transition
    await main_machine.process_event(Event("internal"))
    assert main_machine.current_state == sub2

    # Test boundary transition
    await main_machine.process_event(Event("exit_submachine"))
    assert main_machine.current_state == external


@pytest.mark.asyncio
async def test_async_event_queue_concurrent_operations():
    """Test concurrent operations on AsyncEventQueue."""
    queue = AsyncEventQueue(priority=True)
    events = [Event(f"event{i}", priority=i % 3) for i in range(10)]

    # Enqueue events concurrently
    await asyncio.gather(*(queue.enqueue(event) for event in events))

    # Dequeue events and verify priority ordering
    received_events = []
    while not queue.is_empty():
        event = await queue.dequeue()
        if event:
            received_events.append(event)

    # Verify events were received in priority order
    priorities = [event.priority for event in received_events]
    assert priorities == sorted(priorities, reverse=True)


@pytest.mark.asyncio
async def test_async_state_machine_concurrent_transitions():
    """Test concurrent transitions in AsyncStateMachine."""
    initial = State("initial")
    target1 = State("target1")
    target2 = State("target2")

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target1)
    machine.add_state(target2)

    # Add transitions with different priorities
    machine.add_transition(Transition(source=initial, target=target1, guards=[lambda e: e.name == "test"], priority=1))
    machine.add_transition(Transition(source=initial, target=target2, guards=[lambda e: e.name == "test"], priority=2))

    await machine.start()

    # Process same event multiple times concurrently
    event = Event("test")
    results = await asyncio.gather(*(machine.process_event(event) for _ in range(5)))

    # Verify machine ended up in the higher priority state
    assert machine.current_state == target2
    # At least one transition should have succeeded
    assert any(results)
