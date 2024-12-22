# test_async_trace.py

import asyncio

import pytest

from hsm.core.events import Event
from hsm.core.states import State
from hsm.core.transitions import Transition
from hsm.runtime.async_support import AsyncStateMachine


class AsyncTraceHook:
    def __init__(self):
        self.trace = []

    async def on_enter(self, state):
        self.trace.append(f"ENTER:{state.name}")

    async def on_exit(self, state):
        self.trace.append(f"EXIT:{state.name}")

    # If you want an on_transition or on_error, add them similarly.


@pytest.mark.asyncio
async def test_async_trace():
    """
    Ensures the 'on_enter' and 'on_exit' hooks are called properly in an async context.
    """
    hook = AsyncTraceHook()

    sA = State("A")
    sB = State("B")

    machine = AsyncStateMachine(sA, hooks=[hook])
    machine.add_state(sB)
    machine.add_transition(Transition(sA, sB, guards=[lambda e: e.name == "go"]))

    # Start
    await machine.start()
    # => ["ENTER:A"]

    await machine.process_event(Event("go"))
    # => ["ENTER:A", "EXIT:A", "ENTER:B"]

    # Stop
    await machine.stop()
    # => [..., "EXIT:B"]

    # Check final trace
    # Because it's async, we rely on the order being correct from awaited calls
    # Expected: ["ENTER:A", "EXIT:A", "ENTER:B", "EXIT:B"]
    assert hook.trace == ["ENTER:A", "EXIT:A", "ENTER:B", "EXIT:B"]
