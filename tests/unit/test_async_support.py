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
