# hsm/runtime/event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import heapq
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

from hsm.core.errors import HSMError
from hsm.interfaces.abc import AbstractEvent, AbstractEventQueue
from hsm.interfaces.protocols import Event

logger = logging.getLogger(__name__)


class EventQueueError(HSMError):
    """Base exception for event queue related errors.

    Attributes:
        message: Error description
        details: Additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class QueueFullError(EventQueueError):
    """Raised when attempting to enqueue to a full queue.

    Attributes:
        max_size: Maximum queue capacity
        current_size: Current queue size
    """

    def __init__(self, message: str, max_size: int, current_size: int) -> None:
        super().__init__(message, {"max_size": max_size, "current_size": current_size})
        self.max_size = max_size
        self.current_size = current_size


class QueueEmptyError(EventQueueError):
    """Raised when attempting to dequeue from an empty queue."""

    pass


@dataclass
class QueueStats:
    """Statistics for queue monitoring and debugging."""

    total_enqueued: int = 0
    total_dequeued: int = 0
    max_seen_size: int = 0
    total_wait_time: float = 0.0
    last_operation_time: float = field(default_factory=time.monotonic)

    def record_enqueue(self, queue_size: int) -> None:
        """Record statistics for an enqueue operation."""
        self.total_enqueued += 1
        self.max_seen_size = max(self.max_seen_size, queue_size)
        self.last_operation_time = time.monotonic()

    def record_dequeue(self, wait_time: float) -> None:
        """Record statistics for a dequeue operation."""
        self.total_dequeued += 1
        self.total_wait_time += wait_time
        self.last_operation_time = time.monotonic()


@dataclass(order=False)
class PrioritizedEvent:
    """Wrapper for events that enables priority queue ordering.

    Attributes:
        priority: Event priority (lower numbers = higher priority)
        sequence: Monotonic sequence number for FIFO ordering within same priority
        event: The actual event object
    """

    priority: int
    sequence: int = field(compare=False)
    event: Event = field(compare=False)
    enqueue_time: float = field(default_factory=time.monotonic, compare=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrioritizedEvent):
            return NotImplemented
        # Handle None priorities
        if self.priority is None and other.priority is None:
            return self.sequence == other.sequence
        if self.priority is None or other.priority is None:
            return False
        return (self.priority, self.sequence) == (other.priority, other.sequence)

    def __lt__(self, other: "PrioritizedEvent") -> bool:
        if not isinstance(other, PrioritizedEvent):
            return NotImplemented
        # Handle None priorities - treat None as lowest priority
        if self.priority is None and other.priority is None:
            return self.sequence < other.sequence
        if self.priority is None:
            return False  # None is lowest priority
        if other.priority is None:
            return True  # None is lowest priority
        # Lower priority values should come first (-100 before -99)
        return self.priority < other.priority or (self.priority == other.priority and self.sequence < other.sequence)

    def __le__(self, other: "PrioritizedEvent") -> bool:
        if not isinstance(other, PrioritizedEvent):
            return NotImplemented
        return self < other or self == other


class BaseEventQueue:
    """Base class containing common event queue functionality."""

    def __init__(self, max_size: Optional[int] = None, timeout: Optional[float] = None):
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None")

        self._max_size = max_size
        self._default_timeout = timeout
        self._queue: List[PrioritizedEvent] = []
        self._sequence = 0
        self._stats = QueueStats()
        self._shutdown = False

    def _validate_event(self, event: Event) -> None:
        """Validate event and check shutdown state."""
        if self._shutdown:
            raise EventQueueError("Queue is shut down")

        if not isinstance(event, Event):
            raise TypeError("event must implement Event protocol")

    def _check_full(self) -> None:
        """Check if queue is full and raise appropriate error."""
        if self._max_size is not None and len(self._queue) >= self._max_size:
            raise QueueFullError(
                "Queue is at maximum capacity",
                max_size=self._max_size,
                current_size=len(self._queue),
            )

    def _create_priority_event(self, event: Event) -> PrioritizedEvent:
        """Create a PrioritizedEvent wrapper."""
        priority_event = PrioritizedEvent(
            priority=event.get_priority(),
            sequence=self._sequence,
            event=event,
        )
        self._sequence += 1
        return priority_event

    def _log_enqueue(self, event: Event) -> None:
        """Log enqueue operation."""
        logger.debug(
            "Enqueued event %s (priority=%d, queue_size=%d)",
            event.get_id(),
            event.get_priority(),
            len(self._queue),
        )

    def _log_dequeue(self, event: Event, wait_time: float) -> None:
        """Log dequeue operation."""
        logger.debug(
            "Dequeued event %s (wait_time=%.3fs, queue_size=%d)",
            event.get_id(),
            wait_time,
            len(self._queue),
        )

    def _check_shutdown(self) -> None:
        """Check if queue is shut down and raise appropriate error."""
        if self._shutdown:
            raise EventQueueError("Queue is shut down")

    def _handle_dequeue(self, start_time: float) -> AbstractEvent:
        """Common dequeue logic."""
        if not self._queue:
            raise QueueEmptyError("Queue is empty")

        self._verify_heap()  # Verify heap before dequeue
        event = heapq.heappop(self._queue)
        self._verify_heap()  # Verify heap after dequeue

        wait_time = time.monotonic() - start_time
        self._stats.record_dequeue(wait_time)
        self._log_dequeue(event.event, wait_time)
        return event.event

    def _handle_shutdown(self) -> None:
        """Common shutdown logic."""
        self._shutdown = True

    def _verify_heap(self) -> None:
        """Verify the heap invariant is maintained."""
        for i in range(1, len(self._queue)):
            parent = (i - 1) // 2
            assert self._queue[parent] <= self._queue[i], f"Heap property violated at index {i}"


class EventQueue(BaseEventQueue, AbstractEventQueue):
    """
    Thread-safe priority queue implementation for events.

    This implementation provides:
    - Priority-based ordering (lower numbers = higher priority)
    - FIFO ordering within same priority level
    - Thread-safe operations
    - Optional maximum size limit
    - Non-blocking operations with timeout support

    Runtime Invariants:
    - Thread-safe access to queue operations
    - Priority ordering is maintained
    - FIFO ordering within same priority is preserved
    - Size constraints are enforced
    - Event objects are not modified

    Example:
        queue = EventQueue(max_size=100)
        queue.enqueue(Event("START", priority=1))
        event = queue.dequeue()  # Returns highest priority event
    """

    def __init__(self, max_size: Optional[int] = None, timeout: Optional[float] = 1.0):
        """
        Initialize the event queue.

        Args:
            max_size: Optional maximum queue size (None for unlimited)
            timeout: Default timeout for blocking operations (None for no timeout)

        Raises:
            ValueError: If max_size is not None and <= 0
        """
        super().__init__(max_size, timeout)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def enqueue(self, event: Event) -> None:
        """
        Add an event to the queue.

        Thread-safe operation that maintains priority ordering.

        Args:
            event: Event to enqueue

        Raises:
            QueueFullError: If queue is at max_size
            TypeError: If event is not a valid Event object
        """
        self._validate_event(event)

        with self._lock:
            self._check_full()

            priority_event = self._create_priority_event(event)
            heapq.heappush(self._queue, priority_event)
            self._verify_heap()  # Verify heap after push

            self._stats.record_enqueue(len(self._queue))
            self._not_empty.notify()

            self._log_enqueue(event)
            self._stats.last_operation_time = time.monotonic()

    def dequeue(self) -> AbstractEvent:
        """
        Remove and return the highest priority event.

        Thread-safe operation that maintains priority ordering.

        Returns:
            The next event in priority order

        Raises:
            QueueEmptyError: If queue is empty
            EventQueueError: If queue is shut down
        """
        self._check_shutdown()

        start_time = time.monotonic()
        with self._lock:
            event = self._handle_dequeue(start_time)
            self._not_full.notify()
            return event

    def try_dequeue(self, timeout: Optional[float] = None) -> Optional[AbstractEvent]:
        """
        Try to dequeue an event with an optional timeout.

        Args:
            timeout: Maximum time to wait (None for no wait)

        Returns:
            Event if available, None if timeout or queue is empty
        """
        self._check_shutdown()

        start_time = time.monotonic()
        with self._lock:
            if not self._queue:
                if timeout is None or timeout <= 0:
                    return None
                self._not_empty.wait(timeout=timeout)
                if not self._queue:
                    return None

            event = self._handle_dequeue(start_time)
            self._not_full.notify()
            return event

    def shutdown(self) -> None:
        """
        Shutdown the queue, preventing further operations.

        Any blocked operations will be woken up and raise EventQueueError.
        """
        with self._lock:
            self._handle_shutdown()
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def get_stats(self) -> QueueStats:
        """Get current queue statistics."""
        with self._lock:
            return self._stats

    def is_full(self) -> bool:
        """
        Check if queue is at maximum capacity.

        Returns:
            True if queue is at max_size, False otherwise
        """
        with self._lock:
            return self._max_size is not None and len(self._queue) >= self._max_size

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            True if queue is empty, False otherwise
        """
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> None:
        """Clear all events from the queue."""
        with self._lock:
            self._queue.clear()
            self._not_full.notify_all()

    def size(self) -> int:
        """
        Get current queue size.

        Returns:
            Number of events currently in queue
        """
        with self._lock:
            return len(self._queue)

    def peek(self) -> Optional[AbstractEvent]:
        """
        Look at next event without removing it.

        Returns:
            The next event or None if queue is empty
        """
        with self._lock:
            return self._queue[0].event if self._queue else None


class AsyncEventQueue(BaseEventQueue, AbstractEventQueue):
    """
    Asynchronous event queue implementation.

    This implementation provides the same functionality as EventQueue
    but with async/await support for use with asyncio.

    Runtime Invariants:
    - Thread-safe access to queue operations
    - Priority ordering is maintained
    - FIFO ordering within same priority is preserved
    - Size constraints are enforced
    - Event objects are not modified

    Example:
        queue = AsyncEventQueue(max_size=100)
        await queue.enqueue(Event("START", priority=1))
        event = await queue.dequeue()  # Returns highest priority event
    """

    def __init__(self, max_size: Optional[int] = None, timeout: Optional[float] = None):
        """
        Initialize the async event queue.

        Args:
            max_size: Optional maximum queue size (None for unlimited)
            timeout: Default timeout for blocking operations (None for no timeout)

        Raises:
            ValueError: If max_size is not None and <= 0
        """
        super().__init__(max_size, timeout)
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)

    async def enqueue(self, event: Event) -> None:
        """
        Add an event to the queue asynchronously.

        Args:
            event: Event to enqueue

        Raises:
            QueueFullError: If queue is at max_size
            TypeError: If event is not a valid Event object
        """
        self._validate_event(event)

        async with self._lock:
            # Check if queue is full before waiting
            if self._max_size is not None and len(self._queue) >= self._max_size:
                raise QueueFullError(
                    "Queue is at maximum capacity",
                    max_size=self._max_size,
                    current_size=len(self._queue),
                )

            priority_event = self._create_priority_event(event)
            heapq.heappush(self._queue, priority_event)
            self._verify_heap()  # Verify heap after push

            self._stats.record_enqueue(len(self._queue))
            self._not_empty.notify()

            self._log_enqueue(event)
            self._stats.last_operation_time = time.monotonic()

    async def dequeue(self) -> AbstractEvent:
        """
        Remove and return the highest priority event asynchronously.

        Returns:
            The next event in priority order

        Raises:
            QueueEmptyError: If queue is empty
            EventQueueError: If queue is shut down
        """
        self._check_shutdown()

        start_time = time.monotonic()
        async with self._lock:
            event = self._handle_dequeue(start_time)
            self._not_full.notify()
            return event

    async def try_dequeue(self, timeout: Optional[float] = None) -> Optional[AbstractEvent]:
        """
        Try to dequeue an event with an optional timeout.

        Args:
            timeout: Maximum time to wait (None for no wait)

        Returns:
            Event if available, None if timeout or queue is empty
        """
        self._check_shutdown()

        start_time = time.monotonic()
        async with self._lock:
            if not self._queue:
                if timeout is None or timeout <= 0:
                    return None
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout)
                except asyncio.TimeoutError:
                    return None
                if not self._queue:
                    return None

            event = self._handle_dequeue(start_time)
            self._not_full.notify()
            return event

    async def shutdown(self) -> None:
        """
        Shutdown the queue, preventing further operations.

        Any blocked operations will be woken up and raise EventQueueError.
        """
        async with self._lock:
            self._handle_shutdown()
            self._not_empty.notify_all()
            self._not_full.notify_all()

    async def get_stats(self) -> QueueStats:
        """Get current queue statistics."""
        async with self._lock:
            return self._stats

    async def is_full(self) -> bool:
        """
        Check if queue is at maximum capacity asynchronously.

        Returns:
            True if queue is at max_size, False otherwise
        """
        async with self._lock:
            return self._max_size is not None and len(self._queue) >= self._max_size

    async def is_empty(self) -> bool:
        """
        Check if queue is empty asynchronously.

        Returns:
            True if queue has no events, False otherwise
        """
        async with self._lock:
            return len(self._queue) == 0

    async def clear(self) -> None:
        """Clear all events from the queue asynchronously."""
        async with self._lock:
            self._queue.clear()
            self._not_full.notify_all()

    async def size(self) -> int:
        """
        Get current queue size asynchronously.

        Returns:
            Number of events currently in queue
        """
        async with self._lock:
            return len(self._queue)

    async def peek(self) -> Optional[AbstractEvent]:
        """
        Look at next event without removing it asynchronously.

        Returns:
            The next event or None if queue is empty
        """
        async with self._lock:
            return self._queue[0].event if self._queue else None
