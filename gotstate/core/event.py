"""
Event processing and queue management.

Implements event types, priority-based queuing, and run-to-completion
semantics for the hierarchical state machine.
"""

from __future__ import annotations

import bisect
import threading
import time
import uuid
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import icontract

from gotstate.exceptions import EventQueueFullError, InvalidEventError


class EventKind(Enum):
    """Defines the different types of events in the state machine."""

    SIGNAL = auto()
    CALL = auto()
    TIME = auto()
    CHANGE = auto()
    COMPLETION = auto()


class EventPriority(Enum):
    """Defines priority levels for event processing.

    Numeric values determine queue ordering (lower = higher priority).
    """

    HIGH = 0
    NORMAL = 1
    LOW = 2
    DEFER = 3


@icontract.invariant(lambda self: isinstance(self._kind, EventKind), "Event kind must be valid")
@icontract.invariant(lambda self: isinstance(self._priority, EventPriority), "Event priority must be valid")
@icontract.invariant(lambda self: isinstance(self._event_id, str) and len(self._event_id) > 0, "Event ID must exist")
class Event:
    """Represents an event in the state machine.

    Events are immutable after creation. The event_id is auto-generated
    and unique. Event data (payload) is stored as a read-only dict.

    Class Invariants:
    1. Event kind must be a valid EventKind
    2. Event priority must be a valid EventPriority
    3. Event ID must be a non-empty string
    """

    @icontract.require(lambda kind: isinstance(kind, EventKind), "kind must be a valid EventKind")
    @icontract.require(lambda priority: isinstance(priority, EventPriority), "priority must be a valid EventPriority")
    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "name must be a non-empty string")
    def __init__(
        self,
        name: str,
        kind: EventKind = EventKind.SIGNAL,
        priority: EventPriority = EventPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._event_id = str(uuid.uuid4())
        self._name = name
        self._kind = kind
        self._priority = priority
        self._data: Dict[str, Any] = dict(data) if data else {}
        self._timestamp = time.monotonic()
        self._consumed = False

    @property
    def event_id(self) -> str:
        return self._event_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> EventKind:
        return self._kind

    @property
    def priority(self) -> EventPriority:
        return self._priority

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def is_consumed(self) -> bool:
        return self._consumed

    def consume(self) -> None:
        """Mark the event as consumed."""
        self._consumed = True

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        if self._priority.value != other._priority.value:
            return self._priority.value < other._priority.value
        return self._timestamp < other._timestamp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return self._event_id == other._event_id

    def __hash__(self) -> int:
        return hash(self._event_id)

    def __repr__(self) -> str:
        return f"Event(name={self._name!r}, kind={self._kind.name}, priority={self._priority.name})"


class SignalEvent(Event):
    """Asynchronous signal event with optional payload."""

    def __init__(
        self,
        name: str,
        priority: EventPriority = EventPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, EventKind.SIGNAL, priority, data)


class CallEvent(Event):
    """Synchronous call event with return value support."""

    def __init__(
        self,
        name: str,
        operation: Optional[Callable[..., Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, EventKind.CALL, EventPriority.NORMAL, data)
        self._operation = operation
        self._result: Any = None

    @property
    def operation(self) -> Optional[Callable[..., Any]]:
        return self._operation

    @property
    def result(self) -> Any:
        return self._result

    @result.setter
    def result(self, value: Any) -> None:
        self._result = value


class TimeEvent(Event):
    """Time-based event supporting relative and absolute timing."""

    @icontract.require(lambda duration: duration is None or duration >= 0, "Duration must be non-negative")
    def __init__(
        self,
        name: str,
        duration: Optional[float] = None,
        absolute_time: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, EventKind.TIME, EventPriority.NORMAL, data)
        self._duration = duration
        self._absolute_time = absolute_time
        self._cancelled = False

    @property
    def duration(self) -> Optional[float]:
        return self._duration

    @property
    def absolute_time(self) -> Optional[float]:
        return self._absolute_time

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


class ChangeEvent(Event):
    """Event triggered when a monitored condition changes."""

    @icontract.require(lambda condition: callable(condition), "Condition must be callable")
    def __init__(
        self,
        name: str,
        condition: Callable[[], bool],
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, EventKind.CHANGE, EventPriority.NORMAL, data)
        self._condition = condition
        self._last_value: Optional[bool] = None

    @property
    def condition(self) -> Callable[[], bool]:
        return self._condition

    def evaluate(self) -> bool:
        """Evaluate the condition and return True if it changed to True."""
        current = self._condition()
        changed = current and self._last_value is not True
        self._last_value = current
        return changed


class CompletionEvent(Event):
    """Event generated when a state completes its do-activity or reaches a final state."""

    def __init__(self, name: str, source_state_name: str) -> None:
        super().__init__(name, EventKind.COMPLETION, EventPriority.HIGH)
        self._source_state_name = source_state_name

    @property
    def source_state_name(self) -> str:
        return self._source_state_name


_DEFAULT_MAX_SIZE = 1000


@icontract.invariant(lambda self: self._max_size > 0, "Max size must be positive")
@icontract.invariant(lambda self: len(self._events) <= self._max_size, "Queue must not exceed max size")
class EventQueue:
    """Priority-based event queue with run-to-completion semantics.

    Thread-safe. Events are dequeued in priority order, with FIFO
    ordering within the same priority level.

    Class Invariants:
    1. Max size is always positive
    2. Number of events never exceeds max size
    """

    @icontract.require(lambda max_size: max_size > 0, "max_size must be positive")
    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._max_size = max_size
        self._events: List[Event] = []
        self._deferred: List[Event] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._events) == 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @icontract.require(lambda event: isinstance(event, Event), "Must enqueue an Event instance")
    def enqueue(self, event: Event) -> None:
        """Add an event to the queue, maintaining priority order."""
        with self._lock:
            if len(self._events) + len(self._deferred) >= self._max_size:
                raise EventQueueFullError(f"Event queue is full (max_size={self._max_size})")
            if event.priority == EventPriority.DEFER:
                self._deferred.append(event)
                return
            bisect.insort(self._events, event)

    def dequeue(self) -> Optional[Event]:
        """Remove and return the highest-priority event, or None if empty."""
        with self._lock:
            if self._events:
                return self._events.pop(0)
            return None

    def peek(self) -> Optional[Event]:
        """Return the highest-priority event without removing it."""
        with self._lock:
            if self._events:
                return self._events[0]
            return None

    def clear(self) -> None:
        """Remove all events from the queue."""
        with self._lock:
            self._events.clear()
            self._deferred.clear()

    def flush_deferred(self) -> List[Event]:
        """Move deferred events back into the main queue and return them."""
        with self._lock:
            flushed = list(self._deferred)
            for event in self._deferred:
                bisect.insort(self._events, event)
            self._deferred.clear()
            return flushed

    def cancel(self, event_id: str) -> bool:
        """Cancel an event by its ID. Returns True if found and removed."""
        with self._lock:
            for i, event in enumerate(self._events):
                if event.event_id == event_id:
                    self._events.pop(i)
                    return True
            for i, event in enumerate(self._deferred):
                if event.event_id == event_id:
                    self._deferred.pop(i)
                    return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __repr__(self) -> str:
        return f"EventQueue(size={len(self._events)}, deferred={len(self._deferred)}, max={self._max_size})"
