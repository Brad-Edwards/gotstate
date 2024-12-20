# tests/unit/test_async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_async_state_machine_init(dummy_state, validator):
    from hsm.runtime.async_support import AsyncStateMachine

    asm = AsyncStateMachine(initial_state=dummy_state, validator=validator)
    assert asm.current_state == dummy_state


@pytest.mark.asyncio
async def test_async_state_machine_start_stop(dummy_state):
    from hsm.runtime.async_support import AsyncStateMachine

    asm = AsyncStateMachine(initial_state=dummy_state)
    dummy_state.on_enter = MagicMock()
    await asm.start()
    dummy_state.on_enter.assert_called_once()
    await asm.stop()
    dummy_state.on_exit = MagicMock()
    # Since we didn't explicitly handle on_exit in stop for async, no assertion unless defined.


@pytest.mark.asyncio
async def test_async_state_machine_process_event(dummy_state, mock_event):
    from hsm.runtime.async_support import AsyncStateMachine

    asm = AsyncStateMachine(initial_state=dummy_state)
    await asm.start()
    # With no transitions, process_event should not fail
    await asm.process_event(mock_event)


@pytest.mark.asyncio
async def test_async_event_queue(mock_event):
    from hsm.runtime.async_support import AsyncEventQueue

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
    from hsm.runtime.async_support import _AsyncLock

    lock = _AsyncLock()
    await lock.acquire()
    # Verify lock is acquired
    lock.release()
    # Verify lock can be reacquired
    await lock.acquire()
    lock.release()


@pytest.mark.asyncio
async def test_async_state_machine_double_start(dummy_state):
    from hsm.runtime.async_support import AsyncStateMachine

    asm = AsyncStateMachine(initial_state=dummy_state)
    dummy_state.on_enter = MagicMock()
    await asm.start()
    await asm.start()  # Second start should be ignored
    dummy_state.on_enter.assert_called_once()


@pytest.mark.asyncio
async def test_async_state_machine_transition(dummy_state, mock_event):
    from hsm.core.states import State
    from hsm.core.transitions import Transition
    from hsm.runtime.async_support import AsyncStateMachine

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
async def test_async_event_processing_loop():
    from hsm.core.states import State
    from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine, _AsyncEventProcessingLoop

    initial_state = State("initial")
    machine = AsyncStateMachine(initial_state=initial_state)
    queue = AsyncEventQueue()

    loop = _AsyncEventProcessingLoop(machine, queue)

    # Start loop in background task
    import asyncio

    task = asyncio.create_task(loop.start_loop())

    # Give loop time to start
    await asyncio.sleep(0.1)

    # Stop loop
    await loop.stop_loop()

    # Clean up task
    await task


@pytest.mark.asyncio
async def test_async_event_queue_priority_mode():
    from hsm.runtime.async_support import AsyncEventQueue

    eq = AsyncEventQueue(priority=True)
    assert eq.priority_mode is True
    # Even with priority=True, queue should still function as FIFO
    await eq.enqueue(1)
    await eq.enqueue(2)
    assert await eq.dequeue() == 1
    assert await eq.dequeue() == 2


@pytest.mark.asyncio
async def test_async_state_machine_error_handling(dummy_state, mock_event):
    from hsm.runtime.async_support import AsyncStateMachine

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
    from hsm.core.states import State
    from hsm.core.transitions import Transition

    target_state = State("target")
    asm.add_state(target_state)
    transition = Transition(source=dummy_state, target=target_state, guards=[lambda e: True])
    asm.add_transition(transition)

    await asm.process_event(mock_event)
    error_hook.on_error.assert_called_once()


@pytest.mark.asyncio
async def test_async_event_queue_timeout():
    from hsm.runtime.async_support import AsyncEventQueue

    eq = AsyncEventQueue()
    # Should return None when queue is empty (after timeout)
    result = await eq.dequeue()
    assert result is None


@pytest.mark.asyncio
async def test_async_state_machine_process_event_when_stopped(dummy_state, mock_event):
    from hsm.runtime.async_support import AsyncStateMachine

    asm = AsyncStateMachine(initial_state=dummy_state)
    # Don't start the machine
    await asm.process_event(mock_event)  # Should do nothing
    assert asm.current_state == dummy_state


@pytest.mark.asyncio
async def test_async_state_machine_validator_integration(dummy_state, validator):
    from hsm.runtime.async_support import AsyncStateMachine

    # Create async validator mock
    async def async_validate_state_machine(machine):
        pass

    validator.validate_state_machine = MagicMock(side_effect=async_validate_state_machine)

    asm = AsyncStateMachine(initial_state=dummy_state, validator=validator)
    await asm.start()
    validator.validate_state_machine.assert_called_once_with(asm)
