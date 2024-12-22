# test_nested_trace.py

import pytest

from hsm.core.events import Event
from hsm.core.state_machine import CompositeStateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition


class NestedTraceHook:
    def __init__(self):
        self.trace = []

    def on_enter(self, state):
        self.trace.append(f"ENTER:{state.name}")

    def on_exit(self, state):
        self.trace.append(f"EXIT:{state.name}")


def test_nested_trace():
    """
    A composite 'Top', child composite 'Sub', with leaf states 'Sub1' and 'Sub2'.
    We'll trace the sequence from start up to final transition.
    """
    hook = NestedTraceHook()

    top = CompositeState("Top")
    sub = CompositeState("Sub")
    sub1 = State("Sub1")
    sub2 = State("Sub2")

    # set initial child states
    top.initial_state = sub
    sub.initial_state = sub1

    machine = CompositeStateMachine(top, hooks=[hook])
    machine.add_state(sub, parent=top)
    machine.add_state(sub1, parent=sub)
    machine.add_state(sub2, parent=sub)

    # Transition sub1 -> sub2 on event "toSub2"
    machine.add_transition(Transition(sub1, sub2, guards=[lambda e: e.name == "toSub2"]))

    machine.start()
    # Expected trace so far:
    # ENTER:Top, ENTER:Sub, ENTER:Sub1
    expected_start = ["ENTER:Top", "ENTER:Sub", "ENTER:Sub1"]
    assert hook.trace == expected_start, f"Trace mismatch at start: {hook.trace}"

    # Trigger sub1 -> sub2
    machine.process_event(Event("toSub2"))
    # Now we expect an exit from sub1, then enter sub2
    # So total trace: ["ENTER:Top", "ENTER:Sub", "ENTER:Sub1", "EXIT:Sub1", "ENTER:Sub2"]
    assert len(hook.trace) == 5, f"Trace length mismatch: {hook.trace}"
    assert hook.trace[-2:] == ["EXIT:Sub1", "ENTER:Sub2"]

    # Stop the machine: we expect exit Sub2, exit Sub, exit Top
    machine.stop()
    # => "EXIT:Sub2", "EXIT:Sub", "EXIT:Top"
    assert hook.trace[-3:] == ["EXIT:Sub2", "EXIT:Sub", "EXIT:Top"], f"Final stop trace mismatch: {hook.trace}"
