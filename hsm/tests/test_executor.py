# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Type

import pytest

from hsm.core.actions import NoOpAction
from hsm.core.errors import HSMError
from hsm.core.events import Event
from hsm.core.guards import NoOpGuard
from hsm.core.states import State
from hsm.core.transitions import Transition
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractTransition
from hsm.runtime.executor import Executor, ExecutorContext, ExecutorError, ExecutorState, ExecutorStats, StateChange

# -----------------------------------------------------------------------------
# TEST FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_state() -> State:
    """Fixture providing a basic state implementation."""

    class TestState(State):
        def on_enter(self) -> None:
            pass

        def on_exit(self) -> None:
            pass

    return TestState("test_state")


@pytest.fixture
def mock_states() -> List[State]:
    """Fixture providing a list of test states."""

    class TestState(State):
        def on_enter(self) -> None:
            pass

        def on_exit(self) -> None:
            pass

    return [
        TestState("state1"),
        TestState("state2"),
        TestState("state3"),
    ]


@pytest.fixture
def mock_transitions(mock_states: List[State]) -> List[Transition]:
    """Fixture providing test transitions between states."""
    return [
        Transition(
            source_id=mock_states[0].get_id(),
            target_id=mock_states[1].get_id(),
            guard=NoOpGuard(),
            actions=[NoOpAction()],
            priority=0,
        ),
        Transition(
            source_id=mock_states[1].get_id(),
            target_id=mock_states[2].get_id(),
            guard=NoOpGuard(),
            actions=[NoOpAction()],
            priority=1,
        ),
    ]


@pytest.fixture
def executor(mock_states: List[State], mock_transitions: List[Transition]) -> Executor:
    """Fixture providing a configured executor instance."""
    return Executor(mock_states, mock_transitions, mock_states[0])


# -----------------------------------------------------------------------------
# EXECUTOR CONTEXT TESTS
# -----------------------------------------------------------------------------


def test_executor_context_initialization() -> None:
    """Test ExecutorContext initialization and default values."""
    context = ExecutorContext()
    assert context.state == ExecutorState.IDLE
    assert isinstance(context.get_stats(), ExecutorStats)
    assert context.get_stats().events_processed == 0


def test_executor_context_state_transitions() -> None:
    """Test state transitions in ExecutorContext."""
    context = ExecutorContext()
    context.state = ExecutorState.RUNNING
    assert context.state == ExecutorState.RUNNING
    context.state = ExecutorState.STOPPED
    assert context.state == ExecutorState.STOPPED


def test_executor_context_stats_update() -> None:
    """Test statistics updates in ExecutorContext."""
    context = ExecutorContext()
    context.update_stats(events_processed=1, transitions_executed=2)
    stats = context.get_stats()
    assert stats.events_processed == 1
    assert stats.transitions_executed == 2


def test_executor_context_error_handling() -> None:
    """Test error handler registration and execution."""
    context = ExecutorContext()
    error_handled = False

    def handler(error: Exception) -> None:
        nonlocal error_handled
        error_handled = True

    context.register_error_handler(ValueError, handler)
    context.handle_error(ValueError("test error"))
    assert error_handled


# -----------------------------------------------------------------------------
# EXECUTOR INITIALIZATION TESTS
# -----------------------------------------------------------------------------


def test_executor_initialization(mock_states: List[State], mock_transitions: List[Transition]) -> None:
    """Test Executor initialization with valid configuration."""
    executor = Executor(mock_states, mock_transitions, mock_states[0])
    assert executor._initial_state == mock_states[0]
    assert len(executor._states) == len(mock_states)
    assert len(executor._transitions) == len(mock_transitions)


def test_executor_invalid_initialization(mock_state: State) -> None:
    """Test Executor initialization with invalid configurations."""
    with pytest.raises(ValueError, match="States, transitions, and initial state are required"):
        Executor([], [], None)  # type: ignore

    with pytest.raises(ValueError, match="Initial state must be in states list"):
        Executor([State("other")], [], mock_state)


# -----------------------------------------------------------------------------
# EXECUTOR LIFECYCLE TESTS
# -----------------------------------------------------------------------------


def test_executor_start_stop(executor: Executor) -> None:
    """Test basic start and stop operations."""
    executor.start()
    assert executor.is_running()
    assert executor.get_current_state() is not None

    executor.stop()
    assert not executor.is_running()
    assert executor.get_current_state() is None


def test_executor_double_start(executor: Executor) -> None:
    """Test starting an already running executor."""
    executor.start()
    with pytest.raises(ExecutorError, match="Executor is not in IDLE state"):
        executor.start()
    executor.stop()


def test_executor_stop_when_not_running(executor: Executor) -> None:
    """Test stopping a non-running executor."""
    with pytest.raises(ExecutorError, match="Executor is not running"):
        executor.stop()


# -----------------------------------------------------------------------------
# EVENT PROCESSING TESTS
# -----------------------------------------------------------------------------


def test_event_processing(executor: Executor) -> None:
    """Test basic event processing functionality."""
    executor.start()
    event = Event("test_event")
    executor.process_event(event)
    time.sleep(0.1)  # Allow event processing
    assert executor.get_stats().events_processed > 0
    executor.stop()


def test_event_processing_when_stopped(executor: Executor) -> None:
    """Test event processing when executor is stopped."""
    with pytest.raises(ExecutorError, match="Executor is not running"):
        executor.process_event(Event("test_event"))


def test_state_change_tracking(executor: Executor) -> None:
    """Test state change history tracking."""
    executor.start()
    event = Event("test_event")
    executor.process_event(event)
    time.sleep(0.1)  # Allow state change to complete

    history = executor.get_state_history()
    assert len(history) > 0
    assert isinstance(history[0], StateChange)
    executor.stop()


# -----------------------------------------------------------------------------
# ERROR HANDLING TESTS
# -----------------------------------------------------------------------------


def test_executor_error_handling(executor: Executor) -> None:
    """Test error handling during execution."""
    error_handled = False

    def error_handler(error: Exception) -> None:
        nonlocal error_handled
        error_handled = True

    executor.register_error_handler(ValueError, error_handler)
    executor.start()

    # Simulate error in event processing
    executor._context.handle_error(ValueError("test error"))
    assert error_handled
    executor.stop()


def test_executor_error_inheritance() -> None:
    """Test error class inheritance hierarchy."""
    error = ExecutorError("test")
    assert isinstance(error, HSMError)
    assert isinstance(error, Exception)


def test_executor_error_details() -> None:
    """Test error details preservation."""
    details = {"key": "value"}
    error = ExecutorError("test", details)
    assert error.details == details


# -----------------------------------------------------------------------------
# PAUSE AND RESUME TESTS
# -----------------------------------------------------------------------------


def test_executor_pause_resume(executor: Executor) -> None:
    """Test pause and resume functionality."""
    executor.start()

    with executor.pause():
        assert executor._context.state == ExecutorState.PAUSED

    assert executor._context.state == ExecutorState.RUNNING
    executor.stop()


def test_pause_when_not_running(executor: Executor) -> None:
    """Test pausing when executor is not running."""
    with pytest.raises(ExecutorError, match="Executor is not running"):
        with executor.pause():
            pass


# -----------------------------------------------------------------------------
# STATISTICS AND MONITORING TESTS
# -----------------------------------------------------------------------------


def test_executor_statistics(executor: Executor) -> None:
    """Test statistics collection and reporting."""
    executor.start()
    event = Event("test_event")
    executor.process_event(event)
    time.sleep(0.1)  # Allow event processing

    stats = executor.get_stats()
    assert stats.events_processed > 0
    assert isinstance(stats.avg_processing_time, float)
    assert stats.last_event_time > 0
    executor.stop()


def test_executor_state_history_limit(executor: Executor) -> None:
    """Test state history size limiting."""
    executor.start()
    original_limit = executor._max_state_history
    executor._max_state_history = 2

    # Generate multiple state changes
    for _ in range(5):
        executor.process_event(Event("test_event"))
        time.sleep(0.1)

    history = executor.get_state_history()
    assert len(history) <= 2

    executor._max_state_history = original_limit
    executor.stop()


# -----------------------------------------------------------------------------
# CLEANUP AND RESOURCE MANAGEMENT TESTS
# -----------------------------------------------------------------------------


def test_executor_cleanup(executor: Executor) -> None:
    """Test resource cleanup on shutdown."""
    executor.start()
    executor.stop()

    assert executor._worker_thread is None
    assert executor._current_state is None
    assert executor._context.state == ExecutorState.STOPPED


def test_executor_worker_thread_shutdown(executor: Executor) -> None:
    """Test worker thread shutdown behavior."""
    executor.start()
    thread = executor._worker_thread
    assert thread and thread.is_alive()

    executor.stop()
    time.sleep(0.1)  # Allow thread to stop
    assert not thread.is_alive()


# -----------------------------------------------------------------------------
# TRANSITION AND GUARD TESTS
# -----------------------------------------------------------------------------


def test_transition_with_failing_guard(executor: Executor) -> None:
    """Test transition behavior when guard check fails."""

    class FailingGuard(NoOpGuard):
        def check(self, event: AbstractEvent, state_data: Any) -> bool:
            raise ValueError("Guard check failed")

    executor._transitions[0] = Transition(
        source_id=executor._transitions[0].get_source_state_id(),
        target_id=executor._transitions[0].get_target_state_id(),
        guard=FailingGuard(),
        actions=[NoOpAction()],
        priority=0,
    )

    executor.start()
    event = Event("test_event")
    executor.process_event(event)
    time.sleep(0.1)  # Allow event processing

    # Event should be processed but transition not taken
    stats = executor.get_stats()
    assert stats.events_processed > 0
    assert stats.transitions_executed == 0
    executor.stop()


def test_transition_with_failing_action(executor: Executor) -> None:
    """Test transition behavior when action execution fails."""

    class FailingAction(NoOpAction):
        def execute(self, event: AbstractEvent, state_data: Any) -> None:
            raise ValueError("Action execution failed")

    executor._transitions[0] = Transition(
        source_id=executor._transitions[0].get_source_state_id(),
        target_id=executor._transitions[0].get_target_state_id(),
        guard=NoOpGuard(),
        actions=[FailingAction()],
        priority=0,
    )

    executor.start()
    event = Event("test_event")
    executor.process_event(event)
    time.sleep(0.1)  # Allow event processing

    assert executor._context.state == ExecutorState.ERROR
    # Force stop since we're in error state
    executor._context.state = ExecutorState.RUNNING
    executor.stop()


# -----------------------------------------------------------------------------
# TIMER TESTS
# -----------------------------------------------------------------------------


def test_timer_callback(executor: Executor) -> None:
    """Test timer callback functionality."""
    event_processed = False

    def custom_handler(error: Exception) -> None:
        nonlocal event_processed
        event_processed = True

    executor.register_error_handler(ExecutorError, custom_handler)
    executor.start()

    # Trigger timer callback when not running
    executor._context.state = ExecutorState.STOPPED
    executor._timer._callback("test_timer", Event("timer_event"))
    time.sleep(0.1)

    assert not event_processed  # Should not process event when stopped
    # Reset state to allow clean stop
    executor._context.state = ExecutorState.RUNNING
    executor.stop()


def test_timer_callback_error(executor: Executor) -> None:
    """Test timer callback error handling."""
    error_handled = False

    def error_handler(error: Exception) -> None:
        nonlocal error_handled
        error_handled = True

    executor.register_error_handler(TypeError, error_handler)  # Change to TypeError
    executor.start()

    # Force an error in timer callback
    executor._timer._callback("test_timer", None)  # type: ignore
    time.sleep(0.1)

    assert error_handled
    executor.stop()


# -----------------------------------------------------------------------------
# STATE CHANGE HISTORY TESTS
# -----------------------------------------------------------------------------


def test_state_change_history_attributes(executor: Executor) -> None:
    """Test state change history record attributes."""
    executor.start()
    event = Event("test_event")
    executor.process_event(event)
    time.sleep(0.1)

    history = executor.get_state_history()
    assert len(history) > 0

    change = history[0]
    assert change.source_id == executor._transitions[0].get_source_state_id()
    assert change.target_id == executor._transitions[0].get_target_state_id()
    assert change.event_id == event.get_id()
    assert isinstance(change.timestamp, float)
    assert change.timestamp > 0

    executor.stop()


def test_state_change_history_copy(executor: Executor) -> None:
    """Test that get_state_history returns a copy."""
    executor.start()
    executor.process_event(Event("test_event"))
    time.sleep(0.1)

    history1 = executor.get_state_history()
    history2 = executor.get_state_history()

    assert history1 is not history2  # Should be different objects
    assert history1 == history2  # But with same content
    executor.stop()


# -----------------------------------------------------------------------------
# EXECUTOR CONTEXT THREAD SAFETY TESTS
# -----------------------------------------------------------------------------


def test_executor_context_concurrent_access() -> None:
    """Test thread-safe access to executor context."""
    context = ExecutorContext()
    iterations = 100
    threads = []

    def update_stats() -> None:
        for i in range(iterations):
            context.update_stats(events_processed=i)
            time.sleep(0.001)

    # Create multiple threads updating stats
    for _ in range(3):
        thread = threading.Thread(target=update_stats)
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify stats are consistent
    stats = context.get_stats()
    assert isinstance(stats.events_processed, int)
    assert stats.events_processed >= 0


def test_executor_context_error_handler_thread_safety() -> None:
    """Test thread-safe error handler registration and execution."""
    context = ExecutorContext()
    handler_called = False

    def error_handler(error: Exception) -> None:
        nonlocal handler_called
        handler_called = True
        time.sleep(0.1)  # Simulate work

    def register_and_handle() -> None:
        context.register_error_handler(ValueError, error_handler)
        context.handle_error(ValueError("test"))

    threads = [threading.Thread(target=register_and_handle) for _ in range(3)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert handler_called


# -----------------------------------------------------------------------------
# EDGE CASE TESTS
# -----------------------------------------------------------------------------


def test_executor_stop_timeout(executor: Executor) -> None:
    """Test executor stop behavior with worker thread timeout."""

    class SlowState(State):
        def on_enter(self) -> None:
            pass

        def on_exit(self) -> None:
            time.sleep(2.0)  # Shorter sleep but still longer than timeout

        def get_id(self) -> str:
            return "slow"

    # Create a mock event queue that takes a long time to shutdown
    class SlowEventQueue:
        def __init__(self) -> None:
            self._shutdown = False

        def enqueue(self, event: AbstractEvent) -> None:
            if self._shutdown:
                raise RuntimeError("Queue is shut down")

        def try_dequeue(self, timeout: Optional[float] = None) -> Optional[AbstractEvent]:
            if self._shutdown:
                raise RuntimeError("Queue is shut down")
            return None

        def shutdown(self) -> None:
            time.sleep(2.0)  # Shorter sleep but still longer than timeout
            self._shutdown = True

    # Replace current state and event queue with slow versions
    executor._states = {"slow": SlowState("slow")}
    executor._initial_state = executor._states["slow"]
    executor._event_queue = SlowEventQueue()  # type: ignore

    # Set a very short thread join timeout to force timeout condition
    executor._thread_join_timeout = 0.1  # Add this line to executor class if needed

    executor.start()
    time.sleep(0.1)  # Ensure executor is fully started

    with pytest.raises(ExecutorError, match="Failed to stop executor"):
        executor.stop()

    # Cleanup
    executor._context.state = ExecutorState.STOPPED
    if executor._worker_thread and executor._worker_thread.is_alive():
        executor._worker_thread.join(timeout=0.1)


def test_executor_invalid_target_state(executor: Executor) -> None:
    """Test transition to non-existent target state."""
    # Create transition to non-existent state
    invalid_transition = Transition(
        source_id=executor._transitions[0].get_source_state_id(),
        target_id="non_existent_state",
        guard=NoOpGuard(),
        actions=[NoOpAction()],
        priority=0,
    )
    executor._transitions = [invalid_transition]

    executor.start()
    executor.process_event(Event("test_event"))
    time.sleep(0.1)

    assert executor._context.state == ExecutorState.ERROR
    # Force state to allow clean stop
    executor._context.state = ExecutorState.RUNNING
    executor.stop()


def test_executor_max_queue_size(executor: Executor) -> None:
    """Test executor with maximum queue size limit."""
    executor = Executor(
        states=executor._states.values(),
        transitions=executor._transitions,
        initial_state=executor._initial_state,
        max_queue_size=1,
    )

    executor.start()
    executor.process_event(Event("event1"))

    with pytest.raises(ExecutorError, match="Failed to enqueue event"):
        for _ in range(10):
            executor.process_event(Event("event_overflow"))

    executor.stop()
