# hsm/runtime/timers.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Protocol, Set, Type, Union, runtime_checkable

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

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


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

    def __str__(self) -> str:
        return f"{self.message} (timer_id: {self.timer_id}, state: {self.state})"


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

    def __str__(self) -> str:
        return f"{self.message} (timer_id: {self.timer_id}, duration: {self.duration}, reason: {self.reason})"


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


class BaseTimer(AbstractTimer):
    """Base class for timer implementations.

    Runtime Invariants:
    - Timer state transitions are atomic
    - Callbacks are executed exactly once
    - Cancellation is always possible
    - Resource cleanup is guaranteed
    """

    def __init__(self, timer_id: str, callback: TimerCallback):
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

    def get_info(self) -> TimerInfo:
        """Get current timer information."""
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

    def _cleanup(self) -> None:
        """Clean up timer resources."""
        self._timer = None
        self._event = None
        self._start_time = None
        self._duration = None

    def _validate_schedule(self, duration: float) -> None:
        """Validate schedule parameters and state."""
        if duration < 0:
            raise ValueError("duration cannot be negative")

        if self._state not in (TimerState.IDLE, TimerState.CANCELLED, TimerState.COMPLETED):
            raise TimerSchedulingError(
                f"Timer {self._id} cannot be scheduled in state {self._state}",
                self._id,
                duration,
                "invalid_state",
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

    def _handle_cancel(self, event_id: EventID) -> None:
        """Common cancellation logic."""
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
            raise TimerCancellationError(f"Failed to cancel timer {self._id}: {str(e)}", self._id, self._state) from e

    def _schedule_timer(self, duration: float, event: Event) -> None:
        """Common scheduling setup logic."""
        self._validate_schedule(duration)
        self._start_time = time.time()
        self._duration = duration
        self._event = event
        self._state = TimerState.SCHEDULED

    async def _schedule_timeout_impl(self, duration: float, event: Event) -> None:
        """Template method for scheduling timeouts."""
        self._schedule_timer(duration, event)
        self._create_and_start_timer(duration)

    def _create_and_start_timer(self, duration: float) -> None:
        """Template method for creating and starting a timer."""
        try:
            self._timer = self._create_timer(duration)
            self._start_timer()
            self._state = TimerState.RUNNING
        except Exception as e:
            self._cleanup()
            self._handle_schedule_error(e, duration)

    @abstractmethod
    def _create_timer(self, duration: float) -> Union[threading.Timer, asyncio.TimerHandle]:
        """Create a timer instance."""
        pass

    @abstractmethod
    def _start_timer(self) -> None:
        """Start the timer."""
        pass

    def _handle_schedule_error(self, e: Exception, duration: float) -> None:
        """Common schedule error handling."""
        self._state = TimerState.ERROR
        raise TimerSchedulingError(
            f"Failed to schedule timer {self._id}: {str(e)}", self._id, duration, "scheduling_failed"
        ) from e

    async def _cancel_timeout_impl(self, event_id: EventID) -> None:
        """Template method for cancelling timeouts."""
        self._handle_cancel(event_id)


class Timer(BaseTimer):
    """Thread-safe timer implementation for state machine timeouts.

    Example:
        timer = Timer("timeout_1", callback_fn)
        timer.schedule_timeout(5.0, timeout_event)
        # ... later ...
        timer.cancel_timeout(timeout_event.get_id())
    """

    def __init__(self, timer_id: str, callback: TimerCallback):
        super().__init__(timer_id, callback)
        self._lock = threading.RLock()

    def _create_timer(self, duration: float) -> threading.Timer:
        return threading.Timer(duration, self._on_timeout)

    def _start_timer(self) -> None:
        if isinstance(self._timer, threading.Timer):
            self._timer.start()

    def schedule_timeout(self, duration: float, event: Event) -> None:
        """Schedule a timeout event."""
        with self._lock:
            # Validate duration before attempting async operation
            if duration < 0:
                raise ValueError("duration cannot be negative")

            try:
                asyncio.run(self._schedule_timeout_impl(duration, event))
            except ValueError as e:
                # Re-raise ValueError directly
                raise e
            except Exception as e:
                self._state = TimerState.ERROR
                raise TimerSchedulingError(
                    f"Failed to schedule timer {self._id}: {str(e)}", self._id, duration, "asyncio_error"
                ) from e

    def cancel_timeout(self, event_id: EventID) -> None:
        """Cancel a scheduled timeout."""
        with self._lock:
            try:
                asyncio.run(self._cancel_timeout_impl(event_id))
            except ValueError as e:
                # Re-raise ValueError directly
                raise e
            except Exception as e:
                self._state = TimerState.ERROR
                raise TimerCancellationError(
                    f"Failed to cancel timer {self._id}: {str(e)}", self._id, self._state
                ) from e


class AsyncTimer(BaseTimer):
    """Asynchronous timer implementation for state machine timeouts.

    This class provides the same functionality as Timer but for
    async/await code. It uses asyncio primitives internally.

    Example:
        timer = AsyncTimer("timeout_1", callback_fn)
        await timer.schedule_timeout(5.0, timeout_event)
        # ... later ...
        await timer.cancel_timeout(timeout_event.get_id())
    """

    def __init__(self, timer_id: str, callback: TimerCallback):
        super().__init__(timer_id, callback)
        self._lock = asyncio.Lock()

    def _create_timer(self, duration: float) -> asyncio.TimerHandle:
        loop = asyncio.get_running_loop()
        return loop.call_later(duration, self._on_timeout)

    def _start_timer(self) -> None:
        pass  # asyncio timers start immediately on creation

    async def schedule_timeout(self, duration: float, event: Event) -> None:
        """Schedule a timeout event asynchronously."""
        async with self._lock:
            await self._schedule_timeout_impl(duration, event)

    async def cancel_timeout(self, event_id: EventID) -> None:
        """Cancel a scheduled timeout asynchronously."""
        async with self._lock:
            await self._cancel_timeout_impl(event_id)

    async def shutdown(self) -> None:
        """Shutdown the timer, cancelling any pending timeouts."""
        async with self._lock:
            if self._timer:
                self._timer.cancel()
            self._cleanup()
            self._state = TimerState.IDLE


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

    def _validate_timer_id(self, timer_id: str) -> None:
        """Validate timer ID is unique."""
        if timer_id in self._timers:
            raise ValueError(f"Timer with id {timer_id} already exists")

    def _create_timer_common(
        self, timer_id: str, callback: TimerCallback, timer_class: Type[Union[Timer, AsyncTimer]]
    ) -> Union[Timer, AsyncTimer]:
        """Common timer creation logic."""
        with self._manager_lock:
            self._validate_timer_id(timer_id)
            timer = timer_class(timer_id, callback)
            self._timers[timer_id] = timer
            return timer

    def create_timer(self, timer_id: str, callback: TimerCallback) -> Timer:
        return self._create_timer_common(timer_id, callback, Timer)  # type: ignore

    def create_async_timer(self, timer_id: str, callback: TimerCallback) -> AsyncTimer:
        return self._create_timer_common(timer_id, callback, AsyncTimer)  # type: ignore

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
