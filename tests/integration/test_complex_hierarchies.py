"""Integration tests for complex state hierarchies and error recovery scenarios."""

import asyncio
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock

import pytest

from hsm.core.events import Event, TimeoutEvent
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import CompositeStateMachine
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


def create_nested_state_machine(hook: TestHook) -> CompositeStateMachine:
    """Create a complex nested state machine for testing."""
    # Top level states
    root = CompositeState("Root")
    operational = CompositeState("Operational")
    error = State("Error")
    shutdown = State("Shutdown")

    # Operational substates
    idle = State("Idle")
    processing = CompositeState("Processing")
    paused = State("Paused")

    # Processing substates
    initializing = State("Initializing")
    running = State("Running")
    cleanup = State("Cleanup")

    # Build state hierarchy
    root.add_child_state(operational)
    root.add_child_state(error)
    root.add_child_state(shutdown)

    operational.add_child_state(idle)
    operational.add_child_state(processing)
    operational.add_child_state(paused)

    processing.add_child_state(initializing)
    processing.add_child_state(running)
    processing.add_child_state(cleanup)

    # Create state machines
    processing_machine = AsyncStateMachine(initial_state=initializing, validator=Validator(), hooks=[hook])
    processing_machine.add_state(running)
    processing_machine.add_state(cleanup)

    main_machine = CompositeStateMachine(initial_state=root, validator=Validator(), hooks=[hook])
    main_machine.add_submachine(processing, processing_machine)

    # Add transitions
    def add_transition(source: State, target: State, event_name: str, machine: AsyncStateMachine, priority: int = 0):
        machine.add_transition(
            Transition(source=source, target=target, guards=[lambda e: True], actions=[], priority=priority)
        )

    # Processing machine transitions
    add_transition(initializing, running, "start", processing_machine)
    add_transition(running, cleanup, "finish", processing_machine)
    add_transition(cleanup, initializing, "reset", processing_machine)

    # Main machine transitions
    add_transition(idle, processing, "begin", main_machine)
    add_transition(processing, idle, "complete", main_machine)
    add_transition(operational, error, "error", main_machine, priority=10)
    add_transition(error, operational, "recover", main_machine)
    add_transition(operational, shutdown, "shutdown", main_machine, priority=20)

    return main_machine


@pytest.mark.asyncio
async def test_complex_state_hierarchy(hook):
    """Test navigation through a complex state hierarchy."""
    machine = create_nested_state_machine(hook)
    await machine.start()

    # Verify initial state
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
