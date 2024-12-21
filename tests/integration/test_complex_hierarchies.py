"""Integration tests for complex state hierarchies and error recovery scenarios."""

import asyncio
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from hsm.core.events import Event, TimeoutEvent
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import CompositeStateMachine, StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import ValidationError, Validator
from hsm.runtime.async_support import AsyncEventQueue, AsyncStateMachine


class TestHook:
    """Test hook for tracking state machine events."""

    def __init__(self):
        self.state_changes: List[tuple] = []
        self.errors: List[Exception] = []
        self.action_calls: List[str] = []

    async def on_enter(self, state: State) -> None:
        self.state_changes.append(("enter", state.name))

    async def on_exit(self, state: State) -> None:
        self.state_changes.append(("exit", state.name))

    async def on_error(self, error: Exception) -> None:
        self.errors.append(error)

    async def on_action(self, action_name: str) -> None:
        self.action_calls.append(action_name)


@pytest.fixture
def hook():
    return TestHook()


def create_nested_state_machine(hook):
    """Create a complex nested state machine for testing."""
    # Create root composite state
    root = CompositeState("Root")

    # Create first level states
    operational = CompositeState("Operational")
    error = State("Error")

    # Create second level states under Operational
    processing = CompositeState("Processing")
    idle = State("Idle")

    # Create third level states under Processing
    running = State("Running")
    cleanup = State("Cleanup")

    # Set initial states for composite states
    processing.initial_state = running
    operational.initial_state = idle
    root.initial_state = operational

    # Create state machine with root state
    machine = CompositeStateMachine(initial_state=root, validator=Validator(), hooks=[hook])

    # Add all states to the machine in hierarchical order
    machine.add_state(root)
    machine.add_state(operational, parent=root)
    machine.add_state(error, parent=root)
    machine.add_state(processing, parent=operational)
    machine.add_state(idle, parent=operational)
    machine.add_state(running, parent=processing)
    machine.add_state(cleanup, parent=processing)

    # Build hierarchy (still needed for composite states)
    processing.add_child_state(running)
    processing.add_child_state(cleanup)
    operational.add_child_state(processing)
    operational.add_child_state(idle)
    root.add_child_state(operational)
    root.add_child_state(error)

    # Add transitions for all events
    machine.add_transition(Transition(source=idle, target=processing, guards=[lambda e: e.name == "begin"]))
    machine.add_transition(Transition(source=running, target=cleanup, guards=[lambda e: e.name == "finish"]))
    machine.add_transition(Transition(source=operational, target=error, guards=[lambda e: e.name == "error"]))
    machine.add_transition(Transition(source=error, target=operational, guards=[lambda e: e.name == "recover"]))
    machine.add_transition(Transition(source=running, target=cleanup, guards=[lambda e: e.name == "start"]))

    return machine


@pytest.mark.asyncio
async def test_complex_state_hierarchy(hook):
    """Test navigation through a complex state hierarchy."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    assert machine.current_state.name == "Root"
    assert "Root" in [state for _, state in hook.state_changes]

    # Navigate through the hierarchy
    await machine.process_event(Event("begin"))
    assert machine.current_state.name == "Processing"
    assert "Processing" in [state for _, state in hook.state_changes]

    await machine.process_event(Event("start"))
    assert "Running" in [state for _, state in hook.state_changes]

    await machine.process_event(Event("finish"))
    assert "Cleanup" in [state for _, state in hook.state_changes]


@pytest.mark.asyncio
async def test_error_recovery_scenario(hook):
    """Test error recovery in a complex state hierarchy."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    # Simulate normal operation
    await machine.process_event(Event("begin"))
    await machine.process_event(Event("start"))

    # Simulate error
    await machine.process_event(Event("error"))
    assert machine.current_state.name == "Error"
    assert "Error" in [state for _, state in hook.state_changes]

    # Test recovery
    await machine.process_event(Event("recover"))
    assert machine.current_state.name == "Operational"
    assert "Operational" in [state for _, state in hook.state_changes]


@pytest.mark.asyncio
async def test_concurrent_event_processing(hook):
    """Test concurrent event processing in nested state machines."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    # Create multiple concurrent events
    events = [Event("begin"), Event("start"), Event("error"), Event("recover")]

    # Process events concurrently
    tasks = [asyncio.create_task(machine.process_event(event)) for event in events]

    await asyncio.gather(*tasks)

    # Verify state changes occurred in a valid order
    state_sequence = [state for _, state in hook.state_changes]
    assert "Root" in state_sequence
    assert "Error" in state_sequence
    assert "Operational" in state_sequence


@pytest.mark.asyncio
async def test_shutdown_priority(hook):
    """Test that shutdown events take priority over other transitions."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    # Queue multiple events including shutdown
    events = [Event("begin"), Event("error"), Event("shutdown")]  # Should take priority

    for event in events:
        await machine.process_event(event)

    # Verify we reached shutdown state
    assert machine.current_state.name == "Shutdown"
    assert "Shutdown" in [state for _, state in hook.state_changes]


@pytest.mark.asyncio
async def test_nested_error_handling(hook):
    """Test error handling at different levels of the state hierarchy."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    # Define error-causing action
    async def failing_action(event: Event) -> None:
        raise RuntimeError("Action failed")

    # Add error-prone transition
    machine.add_transition(
        Transition(
            source=machine.get_state("Idle"),
            target=machine.get_state("Processing"),
            guards=[lambda e: True],
            actions=[failing_action],
            priority=0,
        )
    )

    # Attempt transition with failing action
    await machine.process_event(Event("begin"))

    # Verify error was caught and handled
    assert len(hook.errors) == 1
    assert isinstance(hook.errors[0], RuntimeError)
    assert machine.current_state.name == "Error"


@pytest.mark.asyncio
async def test_state_reentry(hook):
    """Test re-entering the same state through different paths."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    # Navigate to processing
    await machine.process_event(Event("begin"))
    initial_processing_entry = len([state for _, state in hook.state_changes if state == "Processing"])

    # Exit and re-enter processing
    await machine.process_event(Event("complete"))
    await machine.process_event(Event("begin"))

    # Verify state was properly re-entered
    processing_entries = len([state for _, state in hook.state_changes if state == "Processing"])
    assert processing_entries == initial_processing_entry + 2  # +2 for exit and re-entry


def test_composite_state_hierarchy():
    """Test hierarchical state structure."""
    # Create child states first
    child1 = State("child1")
    child2 = State("child2")

    # Create root with child1 as initial state
    root = CompositeState("root", initial_state=child1)

    # Create machine with root as initial state
    machine = StateMachine(root)

    # Add states in correct order - parent first, then children
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)

    # Add transition after all states are added
    transition = Transition(source=child1, target=child2)
    machine.add_transition(transition)

    # Verify hierarchy
    assert child1.parent == root
    assert child2.parent == root

    machine.start()
    # The current state should be child1 since it's the initial state of root
    assert machine.current_state == child1
    assert machine.current_state.parent == root

    # Test transition
    event = Event("test")
    assert machine.process_event(event)
    assert machine.current_state == child2
