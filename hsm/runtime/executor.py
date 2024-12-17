# hsm/runtime/executor.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from hsm.core.errors import HSMError
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractTransition
from hsm.interfaces.types import EventID, StateID
from hsm.runtime.event_queue import EventQueue, EventQueueError
from hsm.runtime.timers import Timer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ExecutorError(HSMError):
    """Base exception for executor-related errors.

    Attributes:
        message: Error description
        details: Additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ExecutorState(Enum):
    """Possible states of the executor."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class ExecutorStats:
    """Statistics for monitoring executor performance.

    Attributes:
        events_processed: Total number of events processed
        transitions_executed: Total number of transitions executed
        errors_encountered: Total number of errors encountered
        avg_processing_time: Average time to process an event
        last_event_time: Timestamp of last event processed
    """

    events_processed: int = 0
    transitions_executed: int = 0
    errors_encountered: int = 0
    avg_processing_time: float = 0.0
    last_event_time: float = 0.0


class ExecutorContext:
    """Context for executor operations.

    Manages shared state and provides thread-safe access to executor resources.

    Runtime Invariants:
    - Thread-safe access to shared resources
    - Consistent state transitions
    - Resource cleanup on shutdown
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._state = ExecutorState.IDLE
        self._stats = ExecutorStats()
        self._error_handlers: Dict[Type[Exception], Callable[[Exception], None]] = {}

    @property
    def state(self) -> ExecutorState:
        """Get current executor state."""
        with self._lock:
            return self._state

    @state.setter
    def state(self, new_state: ExecutorState) -> None:
        """Set executor state in a thread-safe manner."""
        with self._lock:
            self._state = new_state

    def update_stats(self, **kwargs: Any) -> None:
        """Update executor statistics."""
        with self._lock:
            current = self._stats.__dict__.copy()
            current.update(kwargs)
            self._stats = ExecutorStats(**current)

    def get_stats(self) -> ExecutorStats:
        """Get current executor statistics."""
        with self._lock:
            return self._stats

    def register_error_handler(self, error_type: Type[Exception], handler: Callable[[Exception], None]) -> None:
        """Register a handler for a specific error type."""
        with self._lock:
            self._error_handlers[error_type] = handler

    def handle_error(self, error: Exception) -> None:
        """Handle an error using registered handlers."""
        with self._lock:
            handler = self._error_handlers.get(type(error))
            if handler:
                handler(error)
            else:
                logger.error("Unhandled error in executor: %s", str(error))
                self.state = ExecutorState.ERROR


@dataclass(frozen=True)
class StateChange:
    """Record of a state change."""

    source_id: StateID
    target_id: StateID
    timestamp: float
    event_id: str


class Executor:
    """
    Synchronous event processor for state machines.

    This class manages the event processing loop, state transitions,
    and error handling for synchronous state machine operations.

    Runtime Invariants:
    - Only one event is processed at a time
    - State transitions are atomic
    - Event processing order follows priority rules
    - Resource cleanup is guaranteed
    - Thread-safe operation

    Example:
        executor = Executor(states, transitions, initial_state)
        executor.start()
        executor.process_event(event)
        executor.stop()
    """

    def __init__(
        self,
        states: List[AbstractState],
        transitions: List[AbstractTransition],
        initial_state: AbstractState,
        max_queue_size: Optional[int] = None,
        thread_join_timeout: float = 1.0,
    ):
        """
        Initialize the executor.

        Args:
            states: List of all possible states
            transitions: List of allowed transitions
            initial_state: Starting state
            max_queue_size: Optional maximum event queue size
            thread_join_timeout: Timeout in seconds for thread join operations (default: 1.0)

        Raises:
            ValueError: If configuration is invalid
        """
        if not states or transitions is None or not initial_state:
            raise ValueError("States, transitions, and initial state are required")
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")
        if thread_join_timeout <= 0:
            raise ValueError("Thread join timeout must be positive")

        self._states = {state.get_id(): state for state in states}
        self._transitions = transitions
        self._initial_state = initial_state
        self._current_state: Optional[AbstractState] = None
        self._event_queue = EventQueue(max_size=max_queue_size)
        self._context = ExecutorContext()
        self._worker_thread: Optional[threading.Thread] = None
        self._state_changes: List[StateChange] = []
        self._max_state_history: int = 1000  # Configurable
        self._thread_join_timeout = thread_join_timeout

        # Initialize timer with default callback
        def timer_callback(timer_id: str, event: AbstractEvent) -> None:
            try:
                if self._context.state == ExecutorState.RUNNING:
                    self.process_event(event)
            except Exception as e:
                logger.error("Timer callback failed: %s", str(e))
                self._context.handle_error(e)

        self._timer = Timer("executor_timer", timer_callback)

    def start(self) -> None:
        """
        Start the executor.

        Initializes the event processing thread and enters the initial state.

        Raises:
            ExecutorError: If executor is already running
        """
        if self._context.state != ExecutorState.IDLE:
            raise ExecutorError("Executor is not in IDLE state")

        try:
            self._current_state = self._initial_state
            self._current_state.on_enter()
            self._context.state = ExecutorState.RUNNING
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            self._context.state = ExecutorState.ERROR
            raise ExecutorError(f"Failed to start executor: {str(e)}") from e

    def stop(self) -> None:
        """
        Stop the executor.

        Exits the current state and stops event processing.

        Raises:
            ExecutorError: If executor is not running or stop operation times out
        """
        if self._context.state != ExecutorState.RUNNING:
            raise ExecutorError("Executor is not running")

        self._context.state = ExecutorState.STOPPING
        stop_start_time = time.time()

        try:
            # Calculate remaining time for operations
            def get_remaining_timeout() -> float:
                elapsed = time.time() - stop_start_time
                remaining = self._thread_join_timeout - elapsed
                if remaining <= 0:
                    raise ExecutorError("Failed to stop executor: operation timed out")
                return remaining

            # Exit current state with timeout
            if self._current_state:
                timeout = get_remaining_timeout()
                exit_thread = threading.Thread(target=self._current_state.on_exit)
                exit_thread.start()
                exit_thread.join(timeout=timeout)
                if exit_thread.is_alive():
                    raise ExecutorError("Failed to stop executor: state exit timed out")

            # Shutdown event queue with timeout
            timeout = get_remaining_timeout()
            queue_thread = threading.Thread(target=self._event_queue.shutdown)
            queue_thread.start()
            queue_thread.join(timeout=timeout)
            if queue_thread.is_alive():
                raise ExecutorError("Failed to stop executor: event queue shutdown timed out")

            # Shutdown timer if available
            if hasattr(self._timer, "shutdown"):
                timeout = get_remaining_timeout()
                timer_thread = threading.Thread(target=self._timer.shutdown)
                timer_thread.start()
                timer_thread.join(timeout=timeout)
                if timer_thread.is_alive():
                    raise ExecutorError("Failed to stop executor: timer shutdown timed out")

            # Join worker thread with remaining timeout
            if self._worker_thread:
                timeout = get_remaining_timeout()
                self._worker_thread.join(timeout=timeout)
                if self._worker_thread.is_alive():
                    raise ExecutorError("Failed to stop executor: worker thread did not terminate")

            self._context.state = ExecutorState.STOPPED

        except Exception as e:
            self._context.state = ExecutorState.ERROR
            raise ExecutorError(f"Failed to stop executor: {str(e)}") from e
        finally:
            self._current_state = None
            self._worker_thread = None

    def process_event(self, event: AbstractEvent) -> None:
        """
        Process an event.

        Thread-safe method to enqueue an event for processing.

        Args:
            event: Event to process

        Raises:
            ExecutorError: If executor is not running
            EventQueueError: If event queue is full
        """
        if self._context.state != ExecutorState.RUNNING:
            raise ExecutorError("Executor is not running")

        try:
            self._event_queue.enqueue(event)
        except EventQueueError as e:
            raise ExecutorError(f"Failed to enqueue event: {str(e)}") from e

    def _process_events(self) -> None:
        """Event processing loop with improved state handling."""
        while self._context.state not in (ExecutorState.STOPPED, ExecutorState.ERROR):
            try:
                if self._context.state == ExecutorState.PAUSED:
                    time.sleep(0.1)  # Reduce CPU usage while paused
                    continue

                if self._context.state != ExecutorState.RUNNING:
                    break

                event = self._event_queue.try_dequeue(timeout=0.1)
                if event:
                    self._handle_event(event)
            except Exception as e:
                logger.exception("Error in event processing loop")
                self._context.handle_error(e)
                if self._context.state == ExecutorState.ERROR:
                    break

    def _handle_event(self, event: AbstractEvent) -> None:
        """Handle a single event."""
        if not self._current_state:
            return

        start_time = time.time()
        try:
            transition = self._find_valid_transition(event)
            if transition:
                self._execute_transition(transition, event)
                self._context.update_stats(transitions_executed=self._context.get_stats().transitions_executed + 1)
        except Exception as e:
            self._context.handle_error(e)
        finally:
            processing_time = time.time() - start_time
            stats = self._context.get_stats()
            self._context.update_stats(
                events_processed=stats.events_processed + 1,
                avg_processing_time=(
                    (stats.avg_processing_time * stats.events_processed + processing_time)
                    / (stats.events_processed + 1)
                ),
                last_event_time=time.time(),
            )

    def _find_valid_transition(self, event: AbstractEvent) -> Optional[AbstractTransition]:
        """Find the highest priority valid transition for the current event."""
        if not self._current_state:
            return None

        valid_transitions = []
        current_state_id = self._current_state.get_id()

        for transition in self._transitions:
            if transition.get_source_state_id() != current_state_id:
                continue

            guard = transition.get_guard()
            if guard:
                try:
                    if guard.check(event, self._current_state.data):
                        valid_transitions.append(transition)
                except Exception as e:
                    logger.error("Guard check failed: %s", str(e))
                    continue
            else:
                valid_transitions.append(transition)

        if not valid_transitions:
            return None

        return max(valid_transitions, key=lambda t: t.get_priority())

    def _execute_transition(self, transition: AbstractTransition, event: AbstractEvent) -> None:
        """Execute a transition."""
        if not self._current_state:
            return

        source_id = transition.get_source_state_id()
        target_id = transition.get_target_state_id()
        target_state = self._states.get(target_id)

        if not target_state:
            raise ExecutorError(f"Invalid target state: {target_id}")

        try:
            self._current_state.on_exit()
            for action in transition.get_actions():
                action.execute(event, self._current_state.data)
            self._current_state = target_state
            self._state_changes.append(
                StateChange(source_id=source_id, target_id=target_id, timestamp=time.time(), event_id=event.get_id())
            )
            if len(self._state_changes) > self._max_state_history:
                self._state_changes = self._state_changes[-self._max_state_history :]
            target_state.on_enter()
        except Exception as e:
            raise ExecutorError(
                f"Transition failed: {str(e)}",
                {
                    "source_state": source_id,
                    "target_state": target_id,
                    "event": event.get_id(),
                },
            ) from e

    @contextmanager
    def pause(self) -> None:
        """
        Temporarily pause event processing.

        Context manager that pauses event processing and resumes on exit.

        Example:
            with executor.pause():
                # Event processing is paused
                do_something()
            # Event processing resumes
        """
        if self._context.state != ExecutorState.RUNNING:
            raise ExecutorError("Executor is not running")

        try:
            self._context.state = ExecutorState.PAUSED
            yield
        finally:
            if self._context.state == ExecutorState.PAUSED:
                self._context.state = ExecutorState.RUNNING

    def get_stats(self) -> ExecutorStats:
        """Get current executor statistics."""
        return self._context.get_stats()

    def register_error_handler(self, error_type: Type[Exception], handler: Callable[[Exception], None]) -> None:
        """Register a custom error handler."""
        self._context.register_error_handler(error_type, handler)

    def get_current_state(self) -> Optional[AbstractState]:
        """Get the current state."""
        return self._current_state

    def is_running(self) -> bool:
        """Check if the executor is running."""
        return self._context.state == ExecutorState.RUNNING

    def get_state_history(self) -> List[StateChange]:
        """Get the state change history."""
        return list(self._state_changes)  # Return copy
