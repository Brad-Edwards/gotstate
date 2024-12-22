# test_complex_hierarchy.py
import threading

import pytest

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.state_machine import CompositeStateMachine, StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition


@pytest.fixture
def complex_hfsm():
    """
    Builds this hierarchy (letters in parentheses are states):

    Root (CompositeState)
    ├─ SubA (CompositeState)
    │   ├─ A1 (State) [initial of SubA]
    │   └─ A2 (State)
    └─ SubB (CompositeState)
        ├─ B1 (State) [initial of SubB]
        └─ B2 (CompositeState)
            ├─ B2a (State) [initial of B2]
            └─ B2b (State)

    Transitions:
    1) A1 -> A2 on event("goA2")
    2) A2 -> B1 on event("switch_to_B")    # crosses from SubA to SubB
    3) B1 -> B2a on event("goB2a")
    4) B2a -> B2b on event("goB2b")
    5) B2b -> A1 on event("reset_to_A1")  # crosses from SubB to SubA

    The Root's initial is SubA, and SubA's initial is A1, SubB's initial is B1,
    and B2's initial is B2a.
    """

    # Create composites
    root = CompositeState("Root")
    subA = CompositeState("SubA")
    subB = CompositeState("SubB")
    subB2 = CompositeState("B2")

    # Create leaf states
    a1 = State("A1")
    a2 = State("A2")
    b1 = State("B1")
    b2a = State("B2a")
    b2b = State("B2b")

    # Set initial children
    subA.initial_state = a1  # subA starts in A1
    subB.initial_state = b1  # subB starts in B1
    subB2.initial_state = b2a

    # Build the top-level composite
    root.initial_state = subA

    # Create machine
    machine = CompositeStateMachine(root)

    # Add states into the hierarchy
    machine.add_state(subA, parent=root)
    machine.add_state(a1, parent=subA)
    machine.add_state(a2, parent=subA)

    machine.add_state(subB, parent=root)
    machine.add_state(b1, parent=subB)
    machine.add_state(subB2, parent=subB)
    machine.add_state(b2a, parent=subB2)
    machine.add_state(b2b, parent=subB2)

    # Add transitions
    # (1) A1 -> A2
    machine.add_transition(Transition(source=a1, target=a2, guards=[lambda e: e.name == "goA2"]))
    # (2) A2 -> B1
    machine.add_transition(Transition(source=a2, target=b1, guards=[lambda e: e.name == "switch_to_B"]))
    # (3) B1 -> B2a
    machine.add_transition(Transition(source=b1, target=b2a, guards=[lambda e: e.name == "goB2a"]))
    # (4) B2a -> B2b
    machine.add_transition(Transition(source=b2a, target=b2b, guards=[lambda e: e.name == "goB2b"]))
    # (5) B2b -> A1
    machine.add_transition(Transition(source=b2b, target=a1, guards=[lambda e: e.name == "reset_to_A1"]))

    return machine


def test_complex_hfsm_initial_states(complex_hfsm):
    """
    Start the machine, ensure it resolves to Root->SubA->A1.
    """
    complex_hfsm.start()
    assert complex_hfsm.current_state is not None
    assert complex_hfsm.current_state.name == "A1", "Expected initial path: Root -> SubA -> A1"


def test_complex_hfsm_nested_transitions(complex_hfsm):
    """
    Verifies transitions from SubA -> SubB and deeper from B1 -> B2 -> ...
    """
    complex_hfsm.start()
    # Currently in A1
    complex_hfsm.process_event(Event("goA2"))
    assert complex_hfsm.current_state.name == "A2"

    # Switch to B1
    complex_hfsm.process_event(Event("switch_to_B"))
    assert complex_hfsm.current_state.name == "B1"

    # Now go deeper in SubB2
    complex_hfsm.process_event(Event("goB2a"))
    assert complex_hfsm.current_state.name == "B2a"

    # Move to B2b
    complex_hfsm.process_event(Event("goB2b"))
    assert complex_hfsm.current_state.name == "B2b"

    # Then reset to A1
    complex_hfsm.process_event(Event("reset_to_A1"))
    assert complex_hfsm.current_state.name == "A1"


def test_complex_hfsm_history_capture(complex_hfsm):
    """
    Demonstrates how a hierarchical HFSM typically handles history:
    If the library supports deep or shallow history, check that re-entry from
    composite returns to the last-active sub-state. If not implemented,
    this test will highlight it (and should fail or skip).
    """
    # Start in A1
    complex_hfsm.start()
    complex_hfsm.process_event(Event("goA2"))  # A1 -> A2
    complex_hfsm.process_event(Event("switch_to_B"))  # A2 -> B1
    assert complex_hfsm.current_state.name == "B1"

    # Stop the machine
    complex_hfsm.stop()

    # Expect that SubA's history is A2
    # If the library doesn't store history, the next start might revert to A1
    # The specification says we do keep history in the StateGraph if configured,
    # so let's see what actually happens:
    subA = complex_hfsm._initial_state.initial_state  # Get SubA from Root's initial state
    hist = complex_hfsm.get_history_state(subA)
    # 'hist' should be None or 'A2' depending on how the code is storing history.
    # If the library implements "record_history" on stop, subA's history was "A2".
    # We'll do a minimal check:
    # If it does not store history properly, we point that out.
    if hist is None:
        pytest.fail("No history was recorded for SubA. HFSM specs often require storing the last sub-state on exit.")
    else:
        assert hist.name == "A2", f"Expected SubA's history to be 'A2', got {hist.name}"

    # Restart
    complex_hfsm.start()

    # Because we are in SubB when we stopped, the parent's history might or might not be stored.
    # This part depends on the implementation details. If the library doesn't fully
    # support multi-level history, you might land back in SubA->A1 instead.
    # The code is not guaranteed to handle multi-level history as-is.
    # We'll see if the machine is in B1 or back in SubA->A1.
    if complex_hfsm.current_state.name != "B1":
        pytest.fail(
            f"Expected the machine to restore to 'B1' via parent's history, but it is in {complex_hfsm.current_state.name}."
        )


def test_complex_hfsm_validation_errors():
    """
    Make sure a composite that has no valid initial state triggers a ValidationError.
    """
    root = CompositeState("Root")
    subX = CompositeState("SubX")
    # No initial_state set for SubX, and no child states.

    # If design strictly requires an initial sub-state for each composite,
    # this should fail on machine.start() or on machine.validate().
    machine = CompositeStateMachine(root)
    machine.add_state(subX, parent=root)
    root.initial_state = subX  # but subX itself is an empty composite

    with pytest.raises(ValidationError):
        machine.start()


def test_concurrent_submachine_start():
    """Test that concurrent submachine initialization is thread-safe."""
    root = CompositeState("Root")
    sub_root = State("SubRoot")
    sub_end = State("SubEnd")

    # Create submachine
    submachine = StateMachine(initial_state=sub_root)
    submachine.add_state(sub_end)
    submachine.add_transition(Transition(source=sub_root, target=sub_end, guards=[lambda e: e.name == "to_sub_end"]))

    # Create main machine
    main_machine = CompositeStateMachine(root)
    main_machine.add_submachine(root, submachine)

    # Create multiple threads trying to start the machine
    threads = []
    for _ in range(5):
        t = threading.Thread(target=main_machine.start)
        threads.append(t)
        t.start()

    # Wait for all threads with timeout
    for t in threads:
        t.join(timeout=2.0)  # Add 2 second timeout
        assert not t.is_alive(), "Thread failed to complete within timeout - possible deadlock"

    # Verify the machine ended up in the correct initial state
    assert (
        main_machine.current_state.name == "SubRoot"
    ), f"Expected to be in SubRoot state, but was in {main_machine.current_state.name}"

    # Try transitioning to verify machine is in valid state
    main_machine.process_event(Event("to_sub_end"))
    assert main_machine.current_state.name == "SubEnd", "Machine failed to transition after concurrent start"
