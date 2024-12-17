# hsm/tests/test_async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
from typing import Any, Dict, List, Optional

import pytest

from hsm.core.errors import HSMError
from hsm.interfaces.abc import AbstractEvent, AbstractState
from hsm.interfaces.types import StateID
from hsm.runtime.async_support import (
    AsyncAction,
    AsyncGuard,
    AsyncHSMError,
    AsyncLockManager,
    AsyncState,
    AsyncStateError,
    AsyncStateMachine,
    AsyncTransition,
    AsyncTransitionError,
)
from hsm.runtime.event_queue import AsyncEventQueue, EventQueueError

# -----------------------------------------------------------------------------
# TEST FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_event() -> AbstractEvent:
    """Fixture providing a mock event for testing."""

    class MockEvent(AbstractEvent):
        def get_id(self) -> str:
            return "test_event"

        def get_payload(self) -> Any:
            return None

        def get_priority(self) -> int:
            return 0

    return MockEvent()


@pytest.fixture
def mock_state() -> AsyncState:
    """Fixture providing a mock async state."""
    return AsyncState("test_state")


@pytest.fixture
def mock_async_component():
    """Factory fixture providing mock async components."""

    class MockGuard(AsyncGuard):
        async def check(self, event: AbstractEvent, state_data: Any) -> bool:
            return True

    class MockAction(AsyncAction):
        async def execute(self, event: AbstractEvent, state_data: Any) -> None:
            # Required by AsyncAction interface and to pass tests
            pass  # NOSONAR

    return {"guard": MockGuard(), "action": MockAction()}


@pytest.fixture
def basic_state_machine(mock_state: AsyncState) -> AsyncStateMachine:
    """Fixture providing a basic async state machine setup."""
    states = [mock_state]
    transitions: List[AsyncTransition] = []
    return AsyncStateMachine(states, transitions, mock_state)


@pytest.fixture
async def running_state_machine(basic_state_machine):
    """Fixture providing a running state machine that auto-stops."""
    await basic_state_machine.start()
    yield basic_state_machine
    await basic_state_machine.stop()


@pytest.fixture
def mock_callback():
    """Fixture providing a state change callback."""
    calls = []

    def callback(source_id: StateID, target_id: StateID):
        calls.append((source_id, target_id))

    callback.calls = calls
    return callback


@pytest.fixture
def error_prone_components():
    """Fixture providing components that raise errors."""

    class ErrorGuard(AsyncGuard):
        async def check(self, event: AbstractEvent, state_data: Any) -> bool:
            raise RuntimeError("Guard error")

    class ErrorAction(AsyncAction):
        async def execute(self, event: AbstractEvent, state_data: Any) -> None:
            raise RuntimeError("Action error")

    return {"guard": ErrorGuard(), "action": ErrorAction()}


# -----------------------------------------------------------------------------
# ERROR HANDLING TESTS
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_class,args,expected_attributes",
    [
        (AsyncHSMError, ["test message"], {"message": "test message"}),
        (
            AsyncStateError,
            ["test message", "test_state", {"key": "value"}],
            {"message": "test message", "state_id": "test_state", "details": {"key": "value"}},
        ),
        (
            AsyncTransitionError,
            ["test message", "source", "target", "mock_event", {"key": "value"}],
            {"message": "test message", "source_id": "source", "target_id": "target", "details": {"key": "value"}},
        ),
    ],
)
def test_error_classes(error_class, args, expected_attributes, mock_event) -> None:
    """Test error classes attributes and inheritance."""
    if error_class == AsyncTransitionError:
        args[3] = mock_event  # Replace "mock_event" string with actual mock_event

    error = error_class(*args)
    assert isinstance(error, HSMError)

    for attr, value in expected_attributes.items():
        assert getattr(error, attr) == value


# -----------------------------------------------------------------------------
# ASYNC STATE TESTS
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_state_initialization() -> None:
    """Test AsyncState initialization and validation."""
    state = AsyncState("test_state")
    assert state.get_id() == "test_state"
    assert isinstance(state.data, dict)
    assert len(state.data) == 0

    with pytest.raises(ValueError):
        AsyncState("")


@pytest.mark.asyncio
async def test_async_state_data_isolation() -> None:
    """Test that state data is properly isolated."""
    state = AsyncState("test_state")
    state.set_data("key", "value")
    data = {"key": "value"}

    # Verify data copy
    assert state.data == data
    assert state.data is not data

    # Verify immutability
    state_data = state.data
    state_data["new_key"] = "new_value"
    assert "new_key" not in state.data


@pytest.mark.asyncio
async def test_async_state_entry_exit() -> None:
    """Test async state entry and exit handlers."""
    state = AsyncState("test_state")

    # Test default implementations
    await state.on_enter()  # Should not raise
    await state.on_exit()  # Should not raise


# -----------------------------------------------------------------------------
# ASYNC TRANSITION TESTS
# -----------------------------------------------------------------------------


def test_async_transition_initialization() -> None:
    """Test AsyncTransition initialization and validation."""
    transition = AsyncTransition("source", "target")
    assert transition.get_source_state_id() == "source"
    assert transition.get_target_state_id() == "target"
    assert transition.get_guard() is None
    assert transition.get_actions() == []
    assert transition.get_priority() == 0

    with pytest.raises(ValueError):
        AsyncTransition("", "target")
    with pytest.raises(ValueError):
        AsyncTransition("source", "")


@pytest.mark.asyncio
async def test_async_transition_with_guard_and_actions(mock_async_component) -> None:
    """Test AsyncTransition with guard and actions."""
    transition = AsyncTransition(
        "source", "target", guard=mock_async_component["guard"], actions=[mock_async_component["action"]], priority=1
    )

    assert transition.get_guard() == mock_async_component["guard"]
    assert transition.get_actions() == [mock_async_component["action"]]
    assert transition.get_priority() == 1


# -----------------------------------------------------------------------------
# ASYNC STATE MACHINE TESTS
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_state_machine_initialization(basic_state_machine: AsyncStateMachine) -> None:
    """Test AsyncStateMachine initialization and validation."""
    assert not basic_state_machine.is_running()
    assert basic_state_machine.get_current_state() is None


@pytest.mark.asyncio
async def test_async_state_machine_start_stop(basic_state_machine: AsyncStateMachine) -> None:
    """Test AsyncStateMachine start and stop operations."""
    # Initial state - not running
    assert not basic_state_machine.is_running()
    assert basic_state_machine.get_current_state() is None

    # After start - running with valid state
    await basic_state_machine.start()
    assert basic_state_machine.is_running()
    assert basic_state_machine.get_current_state() is not None

    # After stop - not running but maintains last state
    await basic_state_machine.stop()
    assert not basic_state_machine.is_running()


@pytest.mark.asyncio
async def test_async_state_machine_event_processing(
    running_state_machine: AsyncStateMachine, mock_event: AbstractEvent
) -> None:
    """Test AsyncStateMachine event processing."""
    await running_state_machine.process_event(mock_event)
    # No need to start/stop explicitly


@pytest.mark.asyncio
async def test_async_state_machine_reset(basic_state_machine: AsyncStateMachine) -> None:
    """Test AsyncStateMachine reset functionality."""
    await basic_state_machine.start()
    assert basic_state_machine.is_running()

    await basic_state_machine.reset()
    assert basic_state_machine.is_running()
    assert basic_state_machine.get_current_state() is not None


@pytest.mark.asyncio
async def test_async_state_machine_context_manager(basic_state_machine: AsyncStateMachine) -> None:
    """Test AsyncStateMachine context manager protocol."""
    async with basic_state_machine as sm:
        assert sm.is_running()
        assert sm.get_current_state() is not None

    assert not basic_state_machine.is_running()


# -----------------------------------------------------------------------------
# ERROR HANDLING AND EDGE CASES
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_state_machine_invalid_states() -> None:
    """Test AsyncStateMachine with invalid state configurations."""
    with pytest.raises(ValueError):
        AsyncStateMachine([], [], AsyncState("test"))

    with pytest.raises(ValueError):
        AsyncStateMachine([AsyncState("test")], [], AsyncState("other"))


@pytest.mark.asyncio
async def test_async_state_machine_event_processing_errors(
    basic_state_machine: AsyncStateMachine, mock_event: AbstractEvent
) -> None:
    """Test error handling during event processing."""
    with pytest.raises(AsyncHSMError):
        await basic_state_machine.process_event(mock_event)  # Should fail when not running


@pytest.mark.asyncio
async def test_async_state_machine_concurrent_operations(
    running_state_machine: AsyncStateMachine, mock_event: AbstractEvent
) -> None:
    """Test concurrent operations on AsyncStateMachine."""
    tasks = [running_state_machine.process_event(mock_event) for _ in range(5)]
    await asyncio.gather(*tasks)


# -----------------------------------------------------------------------------
# VALIDATION AND DEBUGGING
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_state_machine_validation(basic_state_machine: AsyncStateMachine) -> None:
    """Test AsyncStateMachine validation."""
    basic_state_machine.validate()  # Should not raise


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_key", ["current_state", "running", "state_changes"])
async def test_async_state_machine_debug_info(basic_state_machine: AsyncStateMachine, expected_key: str) -> None:
    """Test AsyncStateMachine debug information."""
    debug_info = await basic_state_machine.get_debug_info()
    assert isinstance(debug_info, dict)
    assert expected_key in debug_info


@pytest.mark.parametrize(
    "factory_func,args",
    [
        (AsyncState, [""]),
        (AsyncTransition, ["", "target"]),
        (AsyncTransition, ["source", ""]),
    ],
)
def test_invalid_string_parameters(factory_func, args):
    """Test validation of string parameters across classes."""
    with pytest.raises(ValueError):
        factory_func(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "states,initial_state,expected_error",
    [
        ([], AsyncState("test"), ValueError),
        ([AsyncState("test")], AsyncState("other"), ValueError),
    ],
)
async def test_async_state_machine_invalid_configurations(states, initial_state, expected_error) -> None:
    """Test AsyncStateMachine with invalid configurations."""
    with pytest.raises(expected_error):
        AsyncStateMachine(states, [], initial_state)


async def assert_machine_state(machine: AsyncStateMachine, should_be_running: bool) -> None:
    """Helper to verify state machine state."""
    assert machine.is_running() == should_be_running
    # Only check current state when machine is not running
    if not should_be_running:
        # Remove this assertion since the state machine maintains its last state
        # even when stopped
        pass
    else:
        assert machine.get_current_state() is not None


@pytest.mark.asyncio
async def test_async_lock_manager():
    """Test AsyncLockManager functionality."""
    manager = AsyncLockManager()
    counter = 0

    async def increment():
        nonlocal counter
        await asyncio.sleep(0.1)  # Simulate work
        counter += 1
        return counter

    # Test concurrent operations are properly locked
    tasks = [manager.with_lock("test", increment) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Verify operations were serialized (executed in order)
    assert results == [1, 2, 3, 4, 5]  # Should be sequential


@pytest.mark.asyncio
async def test_state_change_callbacks(basic_state_machine: AsyncStateMachine, mock_callback, mock_event):
    """Test state change callback functionality."""
    basic_state_machine.add_state_change_callback(mock_callback)

    # Create a second state and transition
    second_state = AsyncState("second_state")
    transition = AsyncTransition("test_state", "second_state")

    # Add them to the state machine
    basic_state_machine._states[second_state.get_id()] = second_state
    basic_state_machine._transitions.append(transition)

    await basic_state_machine.start()
    await basic_state_machine.process_event(mock_event)
    await asyncio.sleep(0.1)  # Allow event processing

    assert len(mock_callback.calls) > 0
    assert mock_callback.calls[0] == ("test_state", "second_state")


@pytest.mark.asyncio
async def test_guard_and_action_errors(
    basic_state_machine: AsyncStateMachine, error_prone_components, mock_event: AbstractEvent
):
    """Test error handling in guards and actions."""
    # Add error-prone transition
    transition = AsyncTransition(
        "test_state", "test_state", guard=error_prone_components["guard"], actions=[error_prone_components["action"]]
    )
    basic_state_machine._transitions.append(transition)

    await basic_state_machine.start()
    await basic_state_machine.process_event(mock_event)
    await asyncio.sleep(0.1)  # Allow error to propagate

    # The state machine should still be running despite errors
    assert basic_state_machine.is_running()


@pytest.mark.asyncio
async def test_event_queue_errors(basic_state_machine: AsyncStateMachine, mock_event: AbstractEvent):
    """Test event queue error handling."""
    # Create a full queue scenario
    basic_state_machine._event_queue = AsyncEventQueue(max_size=1)
    await basic_state_machine.start()

    # Fill the queue
    await basic_state_machine.process_event(mock_event)

    # This should raise an error due to full queue
    with pytest.raises(AsyncHSMError):
        await basic_state_machine.process_event(mock_event)


@pytest.mark.asyncio
async def test_debug_info_completeness(basic_state_machine: AsyncStateMachine):
    """Test debug info contains all expected information."""
    await basic_state_machine.start()
    debug_info = await basic_state_machine.get_debug_info()

    expected_keys = {"current_state", "running", "state_changes", "queue_size", "states", "transitions"}

    assert all(key in debug_info for key in expected_keys)
    assert isinstance(debug_info["transitions"], list)
    assert isinstance(debug_info["states"], list)
    assert isinstance(debug_info["queue_size"], int)


@pytest.mark.asyncio
async def test_state_data_operations(mock_state: AsyncState):
    """Test state data operations in detail."""
    # Test setting data
    mock_state.set_data("test_key", "test_value")
    assert mock_state.data["test_key"] == "test_value"

    # Test data immutability
    data = mock_state.data
    data["new_key"] = "new_value"
    assert "new_key" not in mock_state.data

    # Test multiple data operations
    mock_state.set_data("key1", 1)
    mock_state.set_data("key2", 2)
    assert mock_state.data == {"test_key": "test_value", "key1": 1, "key2": 2}


@pytest.mark.asyncio
async def test_transition_validation(basic_state_machine: AsyncStateMachine):
    """Test transition validation."""
    invalid_transition = AsyncTransition("nonexistent", "also_nonexistent")

    with pytest.raises(ValueError):
        basic_state_machine._validate_transition(invalid_transition, basic_state_machine._states)
