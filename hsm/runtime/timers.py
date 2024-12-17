# hsm/runtime/timers.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Protocol, Set, Union, runtime_checkable

from hsm.core.errors import HSMError
from hsm.interfaces.abc import AbstractTimer
from hsm.interfaces.protocols import Event
from hsm.interfaces.types import EventID


class TimerError(HSMError):
    """Base exception for timer-related errors.

    Attributes:
        message: Error description
        details: Additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TimerCancellationError(TimerError):
    """Raised when a timer cannot be cancelled.

    Attributes:
        timer_id: ID of the timer that failed to cancel
        state: Current state of the timer
    """

    def __init__(self, message: str, timer_id: str, state: "TimerState") -> None:
        super().__init__(message, {"timer_id": timer_id, "state": state})
        self.timer_id = timer_id
        self.state = state


class TimerSchedulingError(TimerError):
    """Raised when a timer cannot be scheduled.

    Attributes:
        timer_id: ID of the timer that failed to schedule
        duration: Requested duration
        reason: Reason for scheduling failure
    """

    def __init__(self, message: str, timer_id: str, duration: float, reason: str) -> None:
        super().__init__(message, {"timer_id": timer_id, "duration": duration, "reason": reason})
        self.timer_id = timer_id
        self.duration = duration
        self.reason = reason


class TimerState(Enum):
    """Possible states of a timer."""

    IDLE = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    CANCELLED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass(frozen=True)
class TimerInfo:
    """Information about a timer's current state.

    Attributes:
        id: Unique identifier for the timer
        state: Current state of the timer
        start_time: When the timer was started
        duration: Timer duration in seconds
        remaining: Time remaining (if running)
    """

    id: str
    state: TimerState
    start_time: Optional[float] = None
    duration: Optional[float] = None
    remaining: Optional[float] = None


@runtime_checkable
class TimerCallback(Protocol):
    """Protocol for timer callback functions."""

    def __call__(self, timer_id: str, event: Event) -> None: ...


class Timer(AbstractTimer):
    """
    Thread-safe timer implementation for state machine timeouts.

    Runtime Invariants:
    - Timer state transitions are atomic
    - Callbacks are executed exactly once
    - Cancellation is always possible
    - Resource cleanup is guaranteed

    Example:
        timer = Timer("timeout_1", callback_fn)
        timer.schedule_timeout(5.0, timeout_event)
        # ... later ...
        timer.cancel_timeout(timeout_event.get_id())
    """

    def __init__(self, timer_id: str, callback: TimerCallback):
        """Initialize a new timer.

        Args:
            timer_id: Unique identifier for this timer
            callback: Function to call when timer expires

        Raises:
            ValueError: If timer_id is empty or callback is None
        """
        if not timer_id:
            raise ValueError("timer_id cannot be empty")
        if callback is None:
            raise ValueError("callback cannot be None")

        self._id = timer_id
        self._callback = callback
        self._state = TimerState.IDLE
        self._start_time: Optional[float] = None
        self._duration: Optional[float] = None
        self._timer: Optional[Union[threading.Timer, asyncio.TimerHandle]] = None
        self._event: Optional[Event] = None
        self._lock = threading.RLock()

    def schedule_timeout(self, duration: float, event: Event) -> None:
        """Schedule a timeout event.

        Args:
            duration: Time in seconds until timeout
            event: Event to trigger on timeout

        Raises:
            TimerSchedulingError: If timer cannot be scheduled
            ValueError: If duration is negative
        """
        if duration < 0:
            raise ValueError("duration cannot be negative")

        with self._lock:
            if self._state not in (TimerState.IDLE, TimerState.CANCELLED, TimerState.COMPLETED):
                raise TimerSchedulingError(
                    f"Timer {self._id} cannot be scheduled in state {self._state}",
                    self._id,
                    duration,
                    "invalid_state",
                )

            self._start_time = time.time()
            self._duration = duration
            self._event = event
            self._state = TimerState.SCHEDULED

            try:
                self._timer = threading.Timer(duration, self._on_timeout)
                self._timer.start()
                self._state = TimerState.RUNNING
            except Exception as e:
                self._state = TimerState.ERROR
                raise TimerSchedulingError(
                    f"Failed to schedule timer {self._id}: {str(e)}", self._id, duration, "scheduling_failed"
                ) from e

    def cancel_timeout(self, event_id: EventID) -> None:
        """Cancel a scheduled timeout.

        Args:
            event_id: ID of the event to cancel

        Raises:
            TimerCancellationError: If timer cannot be cancelled
        """
        with self._lock:
            if self._state not in (TimerState.SCHEDULED, TimerState.RUNNING):
                return

            if not self._event or self._event.get_id() != event_id:
                return

            try:
                if self._timer:
                    self._timer.cancel()
                self._state = TimerState.CANCELLED
                self._cleanup()
            except Exception as e:
                self._state = TimerState.ERROR
                raise TimerCancellationError(
                    f"Failed to cancel timer {self._id}: {str(e)}", self._id, self._state
                ) from e

    def get_info(self) -> TimerInfo:
        """Get current timer information.

        Returns:
            TimerInfo object with current state
        """
        with self._lock:
            remaining = None
            if self._state == TimerState.RUNNING and self._start_time and self._duration:
                elapsed = time.time() - self._start_time
                remaining = max(0.0, self._duration - elapsed)

            return TimerInfo(
                id=self._id,
                state=self._state,
                start_time=self._start_time,
                duration=self._duration,
                remaining=remaining,
            )

    def _on_timeout(self) -> None:
        """Handle timer expiration."""
        with self._lock:
            if self._state != TimerState.RUNNING or not self._event:
                return

            try:
                self._callback(self._id, self._event)
                self._state = TimerState.COMPLETED
            except Exception:
                self._state = TimerState.ERROR
            finally:
                self._cleanup()

    def _cleanup(self) -> None:
        """Clean up timer resources."""
        self._timer = None
        self._event = None
        self._start_time = None
        self._duration = None


class AsyncTimer(AbstractTimer):
    """
    Asynchronous timer implementation for state machine timeouts.

    This class provides the same functionality as Timer but for
    async/await code. It uses asyncio primitives internally.

    Runtime Invariants:
    - Timer operations are atomic
    - Callbacks are executed exactly once
    - Cancellation is always possible
    - Resource cleanup is guaranteed
    - Compatible with asyncio event loop

    Example:
        timer = AsyncTimer("timeout_1", callback_fn)
        await timer.schedule_timeout(5.0, timeout_event)
        # ... later ...
        await timer.cancel_timeout(timeout_event.get_id())
    """

    def __init__(self, timer_id: str, callback: TimerCallback):
        """Initialize a new async timer.

        Args:
            timer_id: Unique identifier for this timer
            callback: Function to call when timer expires

        Raises:
            ValueError: If timer_id is empty or callback is None
        """
        if not timer_id:
            raise ValueError("timer_id cannot be empty")
        if callback is None:
            raise ValueError("callback cannot be None")

        self._id = timer_id
        self._callback = callback
        self._state = TimerState.IDLE
        self._start_time: Optional[float] = None
        self._duration: Optional[float] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        self._event: Optional[Event] = None
        self._lock = asyncio.Lock()

    async def schedule_timeout(self, duration: float, event: Event) -> None:
        """Schedule a timeout event asynchronously.

        Args:
            duration: Time in seconds until timeout
            event: Event to trigger on timeout

        Raises:
            TimerSchedulingError: If timer cannot be scheduled
            ValueError: If duration is negative
        """
        if duration < 0:
            raise ValueError("duration cannot be negative")

        async with self._lock:
            if self._state not in (TimerState.IDLE, TimerState.CANCELLED, TimerState.COMPLETED):
                raise TimerSchedulingError(
                    f"Timer {self._id} cannot be scheduled in state {self._state}",
                    self._id,
                    duration,
                    "invalid_state",
                )

            self._start_time = time.time()
            self._duration = duration
            self._event = event
            self._state = TimerState.SCHEDULED

            try:
                loop = asyncio.get_running_loop()
                self._timer = loop.call_later(duration, self._on_timeout)
                self._state = TimerState.RUNNING
            except Exception as e:
                self._state = TimerState.ERROR
                raise TimerSchedulingError(
                    f"Failed to schedule timer {self._id}: {str(e)}", self._id, duration, "scheduling_failed"
                ) from e

    async def cancel_timeout(self, event_id: EventID) -> None:
        """Cancel a scheduled timeout asynchronously.

        Args:
            event_id: ID of the event to cancel

        Raises:
            TimerCancellationError: If timer cannot be cancelled
        """
        async with self._lock:
            if self._state not in (TimerState.SCHEDULED, TimerState.RUNNING):
                return

            if not self._event or self._event.get_id() != event_id:
                return

            try:
                if self._timer:
                    self._timer.cancel()
                self._state = TimerState.CANCELLED
                self._cleanup()
            except Exception as e:
                self._state = TimerState.ERROR
                raise TimerCancellationError(
                    f"Failed to cancel timer {self._id}: {str(e)}", self._id, self._state
                ) from e

    def get_info(self) -> TimerInfo:
        """Get current timer information.

        Returns:
            TimerInfo object with current state
        """
        # No lock needed for read-only operation
        remaining = None
        if self._state == TimerState.RUNNING and self._start_time and self._duration:
            elapsed = time.time() - self._start_time
            remaining = max(0.0, self._duration - elapsed)

        return TimerInfo(
            id=self._id,
            state=self._state,
            start_time=self._start_time,
            duration=self._duration,
            remaining=remaining,
        )

    def _on_timeout(self) -> None:
        """Handle timer expiration."""
        if self._state != TimerState.RUNNING or not self._event:
            return

        try:
            self._callback(self._id, self._event)
            self._state = TimerState.COMPLETED
        except Exception:
            self._state = TimerState.ERROR
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up timer resources."""
        self._timer = None
        self._event = None
        self._start_time = None
        self._duration = None


class TimerManager:
    """
    Central manager for timer creation and tracking.

    This class provides a centralized way to create and manage timers,
    ensuring unique timer IDs and proper cleanup.

    Runtime Invariants:
    - Timer IDs are unique
    - Timer lifecycle is properly managed
    - Thread-safe timer creation and cleanup
    - Proper async/sync timer separation

    Example:
        manager = TimerManager()
        timer = manager.create_timer("timeout_1", callback_fn)
        async_timer = manager.create_async_timer("timeout_2", callback_fn)
    """

    def __init__(self):
        """Initialize the timer manager."""
        self._timers: Dict[str, Union[Timer, AsyncTimer]] = {}
        self._manager_lock = threading.Lock()

    def create_timer(self, timer_id: str, callback: TimerCallback) -> Timer:
        """Create a new synchronous timer.

        Args:
            timer_id: Unique identifier for the timer
            callback: Function to call when timer expires

        Returns:
            New Timer instance

        Raises:
            ValueError: If timer_id already exists
        """
        with self._manager_lock:
            if timer_id in self._timers:
                raise ValueError(f"Timer with id {timer_id} already exists")
            timer = Timer(timer_id, callback)
            self._timers[timer_id] = timer
            return timer

    def create_async_timer(self, timer_id: str, callback: TimerCallback) -> AsyncTimer:
        """Create a new asynchronous timer.

        Args:
            timer_id: Unique identifier for the timer
            callback: Function to call when timer expires

        Returns:
            New AsyncTimer instance

        Raises:
            ValueError: If timer_id already exists
        """
        with self._manager_lock:
            if timer_id in self._timers:
                raise ValueError(f"Timer with id {timer_id} already exists")
            timer = AsyncTimer(timer_id, callback)
            self._timers[timer_id] = timer
            return timer

    def get_timer(self, timer_id: str) -> Optional[Union[Timer, AsyncTimer]]:
        """Get an existing timer by ID.

        Args:
            timer_id: ID of the timer to retrieve

        Returns:
            The timer if it exists, None otherwise
        """
        return self._timers.get(timer_id)

    def remove_timer(self, timer_id: str) -> None:
        """Remove a timer from management.

        Args:
            timer_id: ID of the timer to remove

        Raises:
            ValueError: If timer_id doesn't exist
            TimerError: If timer is still active
        """
        with self._manager_lock:
            if timer_id not in self._timers:
                raise ValueError(f"Timer with id {timer_id} does not exist")

            timer = self._timers[timer_id]
            info = timer.get_info()
            if info.state in (TimerState.SCHEDULED, TimerState.RUNNING):
                raise TimerError(f"Cannot remove active timer {timer_id}")

            del self._timers[timer_id]

    def get_all_timers(self) -> Dict[str, TimerInfo]:
        """Get information about all managed timers.

        Returns:
            Dictionary mapping timer IDs to their current state
        """
        with self._manager_lock:
            return {timer_id: timer.get_info() for timer_id, timer in self._timers.items()}
