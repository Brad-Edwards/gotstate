# test_transition_hook.py

import pytest

from hsm.core.events import Event
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.transitions import Transition


class TransitionTraceHook:
    def __init__(self):
        self.trace = []

    def on_enter(self, state):
        pass  # We'll ignore these for now

    def on_exit(self, state):
        pass

    def on_transition(self, old_state, new_state):
        self.trace.append(f"TRANS:{old_state.name}->{new_state.name}")


def test_transition_trace():
    """
    Verifies that each transition triggers an 'on_transition' hook
    capturing (source->target).
    """
    sA = State("A")
    sB = State("B")
    sC = State("C")

    hook = TransitionTraceHook()
    machine = StateMachine(initial_state=sA, hooks=[hook])
    machine.add_state(sB)
    machine.add_state(sC)

    machine.add_transition(Transition(sA, sB, guards=[lambda e: e.name == "goB"]))
    machine.add_transition(Transition(sB, sC, guards=[lambda e: e.name == "goC"]))

    machine.start()
    machine.process_event(Event("goB"))
    machine.process_event(Event("goC"))

    # We expect two transitions recorded:
    # "TRANS:A->B", "TRANS:B->C"
    assert hook.trace == ["TRANS:A->B", "TRANS:B->C"]
