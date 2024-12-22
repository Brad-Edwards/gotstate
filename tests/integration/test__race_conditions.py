# test_trace.py
import pytest

from gotstate.core.events import Event
from gotstate.core.state_machine import StateMachine
from gotstate.core.states import State
from gotstate.core.transitions import Transition


class TraceHook:
    """
    A hook that appends trace records whenever a state is entered or exited,
    or an error is encountered. By comparing the trace to an expected sequence,
    we can ensure transitions are happening in the correct order.
    """

    def __init__(self, trace_list):
        self.trace = trace_list

    def on_enter(self, state: State):
        self.trace.append(f"ENTER:{state.name}")

    def on_exit(self, state: State):
        self.trace.append(f"EXIT:{state.name}")

    def on_error(self, error: Exception):
        self.trace.append(f"ERROR:{type(error).__name__}")


@pytest.fixture
def traced_machine():
    """
    A simple machine to illustrate capturing trace info.
    States: A -> B -> C
    Transitions triggered by events "toB" and "toC".
    """
    # States
    sA = State("A")
    sB = State("B")
    sC = State("C")

    # We'll add a trace hook in the machine constructor
    trace_log = []
    hook = TraceHook(trace_log)

    # Create the machine
    machine = StateMachine(initial_state=sA, hooks=[hook])
    machine.add_state(sB)
    machine.add_state(sC)

    # Add transitions
    machine.add_transition(Transition(sA, sB, guards=[lambda e: e.name == "toB"]))
    machine.add_transition(Transition(sB, sC, guards=[lambda e: e.name == "toC"]))

    # Return both the machine and the trace_log so test can verify
    return machine, trace_log


def test_trace_linear_path(traced_machine):
    """
    Verifies that the traced sequence of enters/exits is correct
    when going A->B->C.
    """
    machine, trace_log = traced_machine

    # Start machine
    machine.start()
    # At start, on_enter(A) should have fired
    # So trace_log == ["ENTER:A"]

    # Transition A -> B
    machine.process_event(Event("toB"))
    # Expected trace so far:
    # [ "ENTER:A", "EXIT:A", "ENTER:B" ]

    # Transition B -> C
    machine.process_event(Event("toC"))
    # Expected final:
    # [ "ENTER:A", "EXIT:A", "ENTER:B", "EXIT:B", "ENTER:C" ]

    expected = [
        "ENTER:A",
        "EXIT:A",
        "ENTER:B",
        "EXIT:B",
        "ENTER:C",
    ]
    assert trace_log == expected, f"Trace mismatch.\nExpected: {expected}\nGot: {trace_log}"


def test_trace_error_conditions(traced_machine):
    """
    Demonstrates how the trace captures an error. We'll add
    a transition with a faulty action that raises an exception.
    """
    machine, trace_log = traced_machine

    # We'll insert a faulty action in the transition from B->C
    def faulty_action(e):
        raise RuntimeError("Oops")

    # Overwrite the existing transition from B->C with a failing action
    # (We can do so by removing the old transition from the machine's
    # internal graph and then adding our new one if needed.)
    # For simplicity, we assume we can just modify the existing transition's
    # 'actions' directly if your code design allows:
    transitions = machine.get_transitions()
    for t in transitions:
        if t.source.name == "B" and t.target.name == "C":
            t._actions = [faulty_action]

    machine.start()
    # So far: ["ENTER:A"]

    # Transition A->B (OK)
    machine.process_event(Event("toB"))
    # So far: ["ENTER:A", "EXIT:A", "ENTER:B"]

    # Now attempt B->C with the faulty action
    with pytest.raises(RuntimeError, match="Oops"):
        machine.process_event(Event("toC"))

    # The moment an error is raised, the Hook’s on_error() will append:
    # "ERROR:RuntimeError"
    # Meanwhile, “exit_B” and “enter_C” might not complete if the exception
    # halts the transition. That depends on the machine’s design.
    # Let's see the final trace:
    assert "ERROR:RuntimeError" in trace_log, "Trace must record on_error for runtime error."


def test_complex_trace_sequence():
    """
    Illustrates verifying a specific trace pattern, such as A->B->A->B->C,
    to ensure repeated transitions are traced correctly.
    """
    # Build a minimal machine
    sA = State("A")
    sB = State("B")
    sC = State("C")

    trace_log = []
    hook = TraceHook(trace_log)

    machine = StateMachine(sA, hooks=[hook])
    machine.add_state(sB)
    machine.add_state(sC)

    # A->B
    machine.add_transition(Transition(sA, sB, guards=[lambda e: e.name == "goB"]))
    # B->A
    machine.add_transition(Transition(sB, sA, guards=[lambda e: e.name == "goA"]))
    # B->C
    machine.add_transition(Transition(sB, sC, guards=[lambda e: e.name == "goC"]))

    machine.start()
    # => ["ENTER:A"]

    # A->B
    machine.process_event(Event("goB"))
    # => ["ENTER:A", "EXIT:A", "ENTER:B"]

    # B->A
    machine.process_event(Event("goA"))
    # => ["ENTER:A", "EXIT:A", "ENTER:B", "EXIT:B", "ENTER:A"]

    # A->B again
    machine.process_event(Event("goB"))
    # => [..., "EXIT:A", "ENTER:B"]

    # B->C
    machine.process_event(Event("goC"))
    # => [..., "EXIT:B", "ENTER:C"]

    expected_trace = [
        "ENTER:A",
        "EXIT:A",
        "ENTER:B",
        "EXIT:B",
        "ENTER:A",
        "EXIT:A",
        "ENTER:B",
        "EXIT:B",
        "ENTER:C",
    ]
    assert trace_log == expected_trace, f"Trace mismatch.\nExpected: {expected_trace}\nGot: {trace_log}"
