# hsm/runtime/executor.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Type, TypeVar

from hsm.core.errors import ExecutorError
from hsm.core.state_machine import StateMachine
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractTransition
from hsm.interfaces.types import EventID, StateID
from hsm.runtime.event_queue import EventQueue, EventQueueError
from hsm.runtime.timers import Timer

logger = logging.getLogger(__name__)

T = TypeVar("T")
EXECUTOR_IS_NOT_RUNNING = "Executor is not running"
NOT_IDLE_ERROR_MSG = "Executor is not in IDLE state"


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
        executor = Executor(state_machine)
        executor.start()
        executor.process_event(event)
        executor.stop()
    """

    def __init__(
        self,
        state_machine: StateMachine,
        max_queue_size: Optional[int] = None,
        thread_join_timeout: float = 5.0,  # Increased default timeout
    ):
        """
        Initialize the executor.

        Args:
            state_machine: The state machine instance to be managed.
            max_queue_size: Optional maximum event queue size.
            thread_join_timeout: Timeout in seconds for thread join operations (default: 5.0).

        Raises:
            ValueError: If state_machine is None or thread_join_timeout is invalid.
        """
        if state_machine is None:
            raise ValueError("state_machine cannot be None")

        if thread_join_timeout <= 0:
            raise ValueError("Thread join timeout must be positive")

        self._state_machine = state_machine
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

        Starts the state machine and initializes the event processing thread.

        Raises:
            ExecutorError: If executor is not in IDLE state.
        """
        if self._context.state != ExecutorState.IDLE:
            raise ExecutorError(NOT_IDLE_ERROR_MSG)

        try:
            self._state_machine.start()  # Start the state machine
            self._context.state = ExecutorState.RUNNING
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            self._context.state = ExecutorState.ERROR
            raise ExecutorError(f"Failed to start executor: {str(e)}") from e

    def stop(self, force: bool = False) -> None:
        """
        Stop the executor.

        Signals the event processing thread to stop and optionally forces a shutdown.

        Args:
            force: If True, force shutdown even if operations are pending.

        Raises:
            ExecutorError: If executor is not running or if stopping fails.
        """
        if self._context.state != ExecutorState.RUNNING:
            raise ExecutorError(EXECUTOR_IS_NOT_RUNNING)

        self._context.state = ExecutorState.STOPPING
        stop_start_time = time.time()

        try:
            # Calculate remaining time for operations
            def get_remaining_timeout() -> float:
                elapsed = time.time() - stop_start_time
                remaining = self._thread_join_timeout - elapsed
                return remaining

            # Shutdown event queue
            self._event_queue.shutdown()

            # Shutdown timer if available
            if hasattr(self._timer, "shutdown"):
                self._timer.shutdown()

            # Join worker thread
            if self._worker_thread and self._worker_thread.is_alive():
                timeout = get_remaining_timeout()
                if force:
                    self._worker_thread.join(timeout=0)
                else:
                    self._worker_thread.join(timeout=timeout)

                if self._worker_thread.is_alive():
                    if force:
                        logger.warning("Forcing executor shutdown. Some operations might be interrupted.")
                    else:
                        raise ExecutorError("Failed to stop executor: worker thread did not stop within timeout")

            # Stop the state machine only after the worker thread has finished
            self._state_machine.stop()

            self._context.state = ExecutorState.STOPPED

        except Exception as e:
            self._context.state = ExecutorState.ERROR
            raise ExecutorError(f"Failed to stop executor: {str(e)}") from e
        finally:
            self._worker_thread = None

    def process_event(self, event: AbstractEvent) -> None:
        """
        Process an event.

        Enqueues the event for processing by the worker thread.

        Args:
            event: Event to process.

        Raises:
            ExecutorError: If executor is not running or if the event cannot be enqueued.
        """
        if self._context.state != ExecutorState.RUNNING:
            raise ExecutorError(EXECUTOR_IS_NOT_RUNNING)

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
        start_time = time.time()
        try:
            # Send event to the state machine for processing
            self._state_machine.process_event(event)
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

    @contextmanager
    def pause(self) -> Generator[None, None, None]:
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
            raise ExecutorError(EXECUTOR_IS_NOT_RUNNING)

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
        """Get the current state of the state machine."""
        return self._state_machine.get_state()

    def is_running(self) -> bool:
        """Check if the executor is running."""
        return self._context.state == ExecutorState.RUNNING

    def get_state_history(self) -> List[StateChange]:
        """Get the state change history of the state machine."""
        return list(self._state_changes)  # Return copy

    def _join_thread_with_timeout(
        self, thread: threading.Thread, operation: str, timeout: float, start: bool = True
    ) -> None:
        """Join a thread with timeout.

        Args:
            thread: Thread to join
            operation: Description of the operation for error message
            timeout: Maximum time to wait in seconds
            start: Whether to start the thread (default: True)

        Raises:
            ExecutorError: If thread join times out
        """
        if start:
            thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise ExecutorError(f"Failed to stop executor: {operation} timed out")

    def _handle_operation_error(
        self, operation: str, error: Exception, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle operation errors consistently.

        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            details: Optional additional error context
        """
        self._context.state = ExecutorState.ERROR
        raise ExecutorError(f"{operation} failed: {str(error)}", details or {}) from error
