"""Unit tests for the runtime context management."""

from unittest.mock import Mock, call

import pytest

from hsm.core.events import Event
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.runtime.context import RuntimeContext
from hsm.runtime.graph import StateGraph


@pytest.fixture
def setup_context():
    """Create a basic context setup with states and transitions."""
    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")
    state3 = State("state3")  # State with no transitions
    transition = Transition(source=state1, target=state2)

    graph.add_state(state1)
    graph.add_state(state2)
    graph.add_state(state3)
    graph.add_transition(transition)

    context = RuntimeContext(graph, state1)
    return context, state1, state2, transition


def test_initial_state(setup_context):
    """Test that context initializes with correct state."""
    context, state1, _, _ = setup_context
    assert context.get_current_state() == state1


def test_process_event_no_transition(setup_context):
    """Test processing an event with no valid transitions."""
    context, _, _, _ = setup_context
    # Create a new state with no transitions
    state3 = State("state3")
    context._current_state = state3  # Set current state to one with no transitions
    event = Event("unknown")
    assert not context.process_event(event)  # Should return False


def test_process_event_with_transition(setup_context):
    """Test processing an event that triggers a transition."""
    context, state1, state2, transition = setup_context

    # Mock the state methods to verify they're called
    state1.on_exit = Mock()
    state2.on_enter = Mock()
    transition.execute_actions = Mock()

    event = Event("test")
    assert context.process_event(event)  # Should return True

    # Verify the transition sequence
    state1.on_exit.assert_called_once()
    transition.execute_actions.assert_called_once_with(event)
    state2.on_enter.assert_called_once()
    assert context.get_current_state() == state2


def test_composite_state_transitions():
    """Test transitions within composite states."""
    graph = StateGraph()

    # Create composite state hierarchy
    root = CompositeState("root", initial_state=None)
    state1 = State("state1")
    state2 = State("state2")

    graph.add_state(root)
    graph.add_state(state1, parent=root)
    graph.add_state(state2, parent=root)

    transition = Transition(source=state1, target=state2)
    graph.add_transition(transition)

    context = RuntimeContext(graph, state1)

    # Mock methods
    state1.on_exit = Mock()
    state2.on_enter = Mock()
    transition.execute_actions = Mock()

    # Process event
    event = Event("test")
    assert context.process_event(event)

    # Verify correct sequence
    state1.on_exit.assert_called_once()
    transition.execute_actions.assert_called_once_with(event)
    state2.on_enter.assert_called_once()
    assert context.get_current_state() == state2


def test_concurrent_event_processing():
    """Test that concurrent event processing is thread-safe."""
    import queue
    import threading

    graph = StateGraph()
    state1 = State("state1")
    state2 = State("state2")
    state3 = State("state3")

    graph.add_state(state1)
    graph.add_state(state2)
    graph.add_state(state3)

    t1 = Transition(source=state1, target=state2)
    t2 = Transition(source=state1, target=state3)
    graph.add_transition(t1)
    graph.add_transition(t2)

    context = RuntimeContext(graph, state1)
    results = queue.Queue()

    def process_event(event, result_queue):
        success = context.process_event(event)
        result_queue.put((event, success))

    # Create two threads trying to process events concurrently
    event1 = Event("test1")
    event2 = Event("test2")
    thread1 = threading.Thread(target=process_event, args=(event1, results))
    thread2 = threading.Thread(target=process_event, args=(event2, results))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # Collect results
    processed = []
    while not results.empty():
        processed.append(results.get())

    # Only one transition should have succeeded
    assert len([r for _, r in processed if r]) == 1


def test_nested_composite_state_transitions():
    """Test transitions with nested composite states."""
    graph = StateGraph()

    # Create nested structure
    inner_initial = State("inner_initial")
    inner = CompositeState("inner", initial_state=inner_initial)
    outer = CompositeState("outer", initial_state=inner)
    target = State("target")

    graph.add_state(outer)
    graph.add_state(inner, parent=outer)
    graph.add_state(inner_initial, parent=inner)
    graph.add_state(target)

    # Transition from deep nested state to top-level state
    transition = Transition(source=inner_initial, target=target)
    graph.add_transition(transition)

    context = RuntimeContext(graph, inner_initial)

    # Mock methods
    inner_initial.on_exit = Mock()
    target.on_enter = Mock()
    transition.execute_actions = Mock()

    # Process event
    event = Event("test")
    assert context.process_event(event)

    # Verify correct sequence
    inner_initial.on_exit.assert_called_once()
    transition.execute_actions.assert_called_once_with(event)
    target.on_enter.assert_called_once()
    assert context.get_current_state() == target
