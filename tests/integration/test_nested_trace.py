# test_nested_trace.py

import pytest

from hsm.core.events import Event
from hsm.core.state_machine import CompositeStateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator


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
    validator = Validator()

    top = CompositeState("Top")
    sub = CompositeState("Sub")
    sub1 = State("Sub1")
    sub2 = State("Sub2")

    machine = CompositeStateMachine(top, hooks=[hook], validator=validator)

    # Build hierarchy - order matters for proper parent-child relationships
    machine.add_state(sub, parent=top)
    machine.add_state(sub1, parent=sub)
    machine.add_state(sub2, parent=sub)

    validator.validate_state_machine(machine)

    # Set initial states through graph
    machine._graph.set_initial_state(top, sub)
    machine._graph.set_initial_state(sub, sub1)

    machine.add_transition(Transition(sub1, sub2, guards=[lambda e: e.name == "toSub2"]))

    # Start machine - this should trigger entry of all states in hierarchy
    machine.start()

    # Verify initial state entry sequence
    expected_start = ["ENTER:Top", "ENTER:Sub", "ENTER:Sub1"]
    assert hook.trace == expected_start, f"Trace mismatch at start: {hook.trace}"

    machine.process_event(Event("toSub2"))
    assert hook.trace[-2:] == ["EXIT:Sub1", "ENTER:Sub2"]

    machine.stop()
    assert hook.trace[-3:] == ["EXIT:Sub2", "EXIT:Sub", "EXIT:Top"]
