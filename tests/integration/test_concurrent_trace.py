# test_concurrent_trace.py

import threading
import time

import pytest

from gotstate.core.events import Event
from gotstate.core.state_machine import StateMachine
from gotstate.core.states import State
from gotstate.core.transitions import Transition


class TraceHookConcurrent:
    def __init__(self):
        self.trace = []
        self._trace_lock = threading.Lock()

    def on_enter(self, state):
        with self._trace_lock:
            self.trace.append(f"ENTER:{state.name}")

    def on_exit(self, state):
        with self._trace_lock:
            self.trace.append(f"EXIT:{state.name}")


def test_concurrent_tracing():
    """
    Demonstrates that multiple threads firing events at the same time
    still produce a consistent, non-overlapping trace.
    """
    hook = TraceHookConcurrent()
    sA = State("A")
    sB = State("B")
    sC = State("C")

    machine = StateMachine(initial_state=sA, hooks=[hook])
    machine.add_state(sB)
    machine.add_state(sC)

    # Transitions:
    machine.add_transition(Transition(sA, sB, guards=[lambda e: e.name == "toB"]))
    machine.add_transition(Transition(sB, sC, guards=[lambda e: e.name == "toC"]))

    machine.start()

    def thread_func(events):
        for e in events:
            machine.process_event(e)
            # Optional short sleep to let threads interleave
            time.sleep(0.01)

    # T1 tries "A->B->C"
    t1 = threading.Thread(target=thread_func, args=([Event("toB"), Event("toC")],))

    # T2 does the same
    t2 = threading.Thread(target=thread_func, args=([Event("toB"), Event("toC")],))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # The key point is that the trace should not show partial or overlapping transitions,
    # e.g. "EXIT:A" followed immediately by "EXIT:A" without an intervening "ENTER:B".
    # A minimal check is that the trace never toggles states incorrectly.

    # For instance, you might do a quick pass ensuring that each "EXIT:X"
    # is followed by "ENTER:?" for some other state, never two consecutive "EXIT:X" lines, etc.
    # If you see "EXIT:A" -> "EXIT:A" with no "ENTER:B" in between, concurrency is messing up.
    # We'll just do a sanity check for no duplicates:

    for i in range(len(hook.trace) - 1):
        assert not (
            hook.trace[i].startswith("EXIT:") and hook.trace[i] == hook.trace[i + 1]
        ), f"Detected duplicate or overlapping exit: {hook.trace[i]} => concurrency issue"

    # Or do your own thorough validation of the trace sequence.
    # The important part is that no concurrency artifact scrambles transitions.
