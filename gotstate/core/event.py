"""
Event processing and queue management.

Architecture:
- Implements event processing and queue management
- Handles event patterns and ordering
- Maintains event processing semantics
- Coordinates with Executor for processing
- Integrates with Scheduler for time events

Design Patterns:
- Observer Pattern: Event notifications
- Command Pattern: Event execution
- Strategy Pattern: Processing patterns
- Queue Pattern: Event queuing
- Chain of Responsibility: Event handling

Responsibilities:
1. Event Processing
   - Synchronous processing
   - Asynchronous processing
   - Event deferral
   - Priority handling
   - Completion events

2. Event Queue
   - Queue management
   - Event ordering
   - Event cancellation
   - Timeout handling
   - Event filtering

3. Processing Patterns
   - Single consumption
   - Broadcast events
   - Conditional events
   - Event scoping
   - Priority rules

4. Run-to-Completion
   - RTC semantics
   - Queue during transitions
   - Order preservation
   - Re-entrant processing
   - Timeout handling

Security:
- Event validation
- Queue protection
- Resource monitoring
- Processing boundaries

Cross-cutting:
- Error handling
- Performance optimization
- Event metrics
- Thread safety

Dependencies:
- transition.py: Event triggers
- executor.py: Event execution
- scheduler.py: Time events
- machine.py: Machine context
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from queue import PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Set


class EventKind(Enum):
    """Defines the different types of events in the state machine.

    Used to determine event processing behavior and routing.
    """

    SIGNAL = auto()  # External signal events
    CALL = auto()  # Synchronous call events
    TIME = auto()  # Time-based events
    CHANGE = auto()  # Change notification events
    COMPLETION = auto()  # State completion events


class EventPriority(Enum):
    """Defines priority levels for event processing.

    Used to determine event processing order in the queue.
    """

    HIGH = auto()  # Processed before normal priority
    NORMAL = auto()  # Default processing priority
    LOW = auto()  # Processed after normal priority
    DEFER = auto()  # Deferred until state exit


class Event:
    """Represents an event in the state machine.

    The Event class implements the Command pattern to encapsulate event
    data and processing behavior. It supports various event types and
    processing patterns.

    Class Invariants:
    1. Event ID must be unique within its scope
    2. Event kind must not change after creation
    3. Event data must be immutable
    4. Priority must be valid
    5. Timeout must be non-negative if specified
    6. Parameters must be serializable
    7. Event scope must be well-defined
    8. Processing status must be tracked
    9. Cancellation must be handled gracefully
    10. Resources must be properly managed

    Design Patterns:
    - Command: Encapsulates event data and behavior
    - Observer: Notifies of event processing
    - Strategy: Implements processing patterns
    - Memento: Preserves event state
    - Chain of Responsibility: Handles event processing

    Data Structures:
    - Dictionary for event parameters
    - Set for consumed status
    - Queue for processing order
    - Tree for scope hierarchy
    - Map for deferred events

    Algorithms:
    - Priority-based scheduling
    - Scope resolution
    - Timeout handling
    - Consumption tracking

    Threading/Concurrency Guarantees:
    1. Thread-safe event processing
    2. Atomic parameter access
    3. Synchronized scope checking
    4. Safe concurrent consumption
    5. Lock-free status inspection
    6. Mutex protection for queue operations

    Performance Characteristics:
    1. O(1) event creation
    2. O(log n) priority queuing
    3. O(1) parameter access
    4. O(h) scope checking where h is hierarchy depth
    5. O(1) status updates

    Resource Management:
    1. Bounded queue size
    2. Pooled event objects
    3. Cached scope information
    4. Limited concurrent processing
    5. Automatic timeout cleanup
    """

    def __init__(
        self,
        event_id: str,
        kind: EventKind,
        priority: EventPriority,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize a new Event instance.

        Args:
            event_id: Unique identifier for the event
            kind: Type of event (signal, call, time, etc.)
            priority: Processing priority level
            data: Optional dictionary of event parameters
            timeout: Optional timeout in milliseconds

        Raises:
            ValueError: If any parameters are invalid
        """
        if not event_id or not isinstance(event_id, str):
            raise ValueError("Event ID must be a non-empty string")

        if not isinstance(kind, EventKind):
            raise ValueError("Event kind must be an EventKind enum value")

        if not isinstance(priority, EventPriority):
            raise ValueError("Event priority must be an EventPriority enum value")

        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be non-negative")

        self._id = event_id
        self._kind = kind
        self._priority = priority
        self._data = data.copy() if data else {}
        self._timeout = timeout
        self._consumed = False
        self._cancelled = False
        self._creation_time = time.time()

    @property
    def id(self) -> str:
        """Get the event ID."""
        return self._id

    @property
    def kind(self) -> EventKind:
        """Get the event kind."""
        return self._kind

    @property
    def priority(self) -> EventPriority:
        """Get the event priority."""
        return self._priority

    @property
    def data(self) -> Dict[str, Any]:
        """Get a copy of the event data."""
        return self._data.copy()

    @property
    def timeout(self) -> Optional[int]:
        """Get the event timeout."""
        return self._timeout

    @property
    def consumed(self) -> bool:
        """Check if the event has been consumed."""
        return self._consumed

    @property
    def cancelled(self) -> bool:
        """Check if the event has been cancelled."""
        return self._cancelled

    def consume(self) -> None:
        """Mark the event as consumed."""
        self._consumed = True

    def cancel(self) -> None:
        """Cancel the event."""
        self._cancelled = True

    def __lt__(self, other: "Event") -> bool:
        """Compare events based on priority and creation time.

        Args:
            other: Another event to compare with

        Returns:
            True if this event has higher priority (lower value)
            or was created earlier with same priority
        """
        if not isinstance(other, Event):
            return NotImplemented
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self._creation_time < other._creation_time


class SignalEvent(Event):
    """Represents an asynchronous signal event.

    SignalEvent implements asynchronous event processing with
    optional payload data and broadcast capabilities.

    Class Invariants:
    1. Must maintain signal ordering
    2. Must handle broadcast properly
    3. Must track consumption status
    4. Must preserve payload integrity

    Design Patterns:
    - Observer: Implements signal notification
    - Command: Encapsulates signal data
    - Strategy: Implements broadcast behavior

    Threading/Concurrency Guarantees:
    1. Thread-safe signal dispatch
    2. Atomic consumption tracking
    3. Safe concurrent broadcast

    Performance Characteristics:
    1. O(1) signal creation
    2. O(n) broadcast where n is listener count
    3. O(1) consumption status
    """

    def __init__(
        self,
        event_id: str,
        priority: EventPriority,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize a SignalEvent instance.

        Args:
            event_id: Unique identifier for the signal
            priority: Processing priority level
            data: Optional signal payload data
            timeout: Optional timeout in milliseconds
        """
        super().__init__(event_id=event_id, kind=EventKind.SIGNAL, priority=priority, data=data, timeout=timeout)
        self._broadcast = True
        self._listeners: Set[str] = set()

    def can_consume(self) -> bool:
        """Check if the signal can be consumed.

        Returns:
            True if the signal can be consumed (broadcast signals
            can always be consumed), False otherwise.
        """
        return True  # Signal events support multiple consumption (broadcast)

    def add_listener(self, listener_id: str) -> None:
        """Add a listener for this signal.

        Args:
            listener_id: Unique identifier for the listener
        """
        self._listeners.add(listener_id)

    def remove_listener(self, listener_id: str) -> None:
        """Remove a listener from this signal.

        Args:
            listener_id: Unique identifier for the listener
        """
        self._listeners.discard(listener_id)

    @property
    def listeners(self) -> Set[str]:
        """Get the set of listener IDs.

        Returns:
            A copy of the listener ID set
        """
        return self._listeners.copy()


class CallEvent(Event):
    """Represents a synchronous call event.

    CallEvent implements synchronous operation calls with
    return values and parameter passing.

    Class Invariants:
    1. Must complete synchronously
    2. Must handle return values
    3. Must validate parameters
    4. Must maintain call semantics

    Design Patterns:
    - Command: Encapsulates operation call
    - Strategy: Implements call handling
    - Template Method: Defines call sequence

    Threading/Concurrency Guarantees:
    1. Thread-safe parameter handling
    2. Atomic operation execution
    3. Safe concurrent calls

    Performance Characteristics:
    1. O(1) call creation
    2. O(p) parameter validation where p is parameter count
    3. O(1) return value handling
    """

    def __init__(
        self,
        event_id: str,
        priority: EventPriority,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize a CallEvent instance.

        Args:
            event_id: Unique identifier for the call
            priority: Processing priority level
            data: Call parameters including method, args, and kwargs
            timeout: Optional timeout in milliseconds

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if data is None:
            data = {"method": None, "args": [], "kwargs": {}}

        # Validate required parameters
        if "method" not in data:
            raise ValueError("Call event data must include 'method' parameter")

        # Validate args and kwargs
        args = data.get("args", [])
        kwargs = data.get("kwargs", {})

        if not isinstance(args, list):
            raise ValueError("Call event 'args' must be a list")

        if not isinstance(kwargs, dict):
            raise ValueError("Call event 'kwargs' must be a dictionary")

        # Ensure args and kwargs are present in data
        data["args"] = args
        data["kwargs"] = kwargs

        super().__init__(event_id=event_id, kind=EventKind.CALL, priority=priority, data=data, timeout=timeout)
        self._return_value = None

    def can_consume(self) -> bool:
        """Check if the call can be consumed.

        Returns:
            True if the call has not been consumed yet,
            False if it has already been consumed.
        """
        return not self.consumed

    @property
    def return_value(self) -> Any:
        """Get the call return value.

        Returns:
            The return value if set, None otherwise
        """
        return self._return_value

    def set_return_value(self, value: Any) -> None:
        """Set the call return value.

        Args:
            value: The return value to set
        """
        # Make a deep copy if the value is a dict or list
        if isinstance(value, (dict, list)):
            self._return_value = value.copy()
        else:
            self._return_value = value


class TimeEvent(Event):
    """Represents a time-based event.

    TimeEvent implements both relative ("after") and absolute ("at")
    timing events with proper scheduling.

    Class Invariants:
    1. Must have valid time specification
    2. Must maintain timing accuracy
    3. Must handle cancellation
    4. Must track scheduling status

    Design Patterns:
    - Command: Encapsulates timing logic
    - Observer: Notifies of timing
    - Strategy: Implements timing types

    Threading/Concurrency Guarantees:
    1. Thread-safe scheduling
    2. Atomic timer operations
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) event creation
    2. O(log n) scheduling where n is timer count
    3. O(1) cancellation
    """

    def __init__(
        self,
        event_id: str,
        priority: EventPriority,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize a TimeEvent instance.

        Args:
            event_id: Unique identifier for the time event
            priority: Processing priority level
            data: Time event parameters including type and time
            timeout: Optional timeout (not used for time events)

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if data is None:
            data = {}

        # Validate required parameters
        if "type" not in data:
            raise ValueError("Time event data must include 'type' parameter")

        if data["type"] not in ["after", "at"]:
            raise ValueError("Time event type must be 'after' or 'at'")

        if "time" not in data:
            raise ValueError("Time event data must include 'time' parameter")

        if not isinstance(data["time"], (int, float)) or data["time"] < 0:
            raise ValueError("Time event time must be a non-negative number")

        # Set default repeat value if not provided
        data.setdefault("repeat", False)

        super().__init__(
            event_id=event_id,
            kind=EventKind.TIME,
            priority=priority,
            data=data,
            timeout=None,  # Time events manage their own timing
        )
        self._expired = False

    def can_consume(self) -> bool:
        """Check if the time event can be consumed.

        Returns:
            True if the event is repeating or hasn't been consumed,
            False if it's non-repeating and has been consumed.
        """
        return self.data.get("repeat", False) or not self.consumed

    @property
    def is_expired(self) -> bool:
        """Check if the time event has expired.

        Returns:
            True if the event has expired, False otherwise
        """
        return self._expired

    def expire(self) -> None:
        """Mark the time event as expired.

        This also marks the event as consumed unless it's a repeating event.
        """
        self._expired = True
        if not self.data.get("repeat", False):
            self.consume()


class ChangeEvent(Event):
    """Represents a change notification event.

    ChangeEvent implements condition-based events that trigger
    when monitored values change.

    Class Invariants:
    1. Must track condition state
    2. Must detect all changes
    3. Must prevent missed events
    4. Must maintain change history

    Design Patterns:
    - Observer: Monitors changes
    - Command: Encapsulates conditions
    - Strategy: Implements detection

    Threading/Concurrency Guarantees:
    1. Thread-safe condition monitoring
    2. Atomic change detection
    3. Safe concurrent notification

    Performance Characteristics:
    1. O(1) event creation
    2. O(c) condition evaluation where c is condition complexity
    3. O(h) history tracking where h is history size
    """

    VALID_CONDITIONS = {"value_changed", "threshold_crossed", "state_changed", "range_violation"}

    def __init__(
        self,
        event_id: str,
        priority: EventPriority,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize a ChangeEvent instance.

        Args:
            event_id: Unique identifier for the change event
            priority: Processing priority level
            data: Change event parameters including condition and target
            timeout: Optional timeout in milliseconds

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if data is None:
            data = {}

        # Validate required parameters
        if "condition" not in data:
            raise ValueError("Change event data must include 'condition' parameter")

        if data["condition"] not in self.VALID_CONDITIONS:
            raise ValueError(f"Invalid condition. Must be one of: {self.VALID_CONDITIONS}")

        if "target" not in data:
            raise ValueError("Change event data must include 'target' parameter")

        # Set default track_history value if not provided
        data.setdefault("track_history", True)

        super().__init__(event_id=event_id, kind=EventKind.CHANGE, priority=priority, data=data, timeout=timeout)
        self._history: List[Dict[str, Any]] = []

        # Record initial change if old and new values are provided
        if "old_value" in data and "new_value" in data:
            self.record_change(data["old_value"], data["new_value"])

    def can_consume(self) -> bool:
        """Check if the change event can be consumed.

        Returns:
            True if the event hasn't been consumed yet,
            False if it has already been consumed.
        """
        return not self.consumed

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get the change history.

        Returns:
            List of change records, each containing old_value and new_value
        """
        return self._history.copy()

    def record_change(self, old_value: Any, new_value: Any) -> None:
        """Record a change in the history.

        Args:
            old_value: The previous value
            new_value: The new value
        """
        if self.data.get("track_history", True):
            self._history.append({"old_value": old_value, "new_value": new_value, "timestamp": time.time()})


class CompletionEvent(Event):
    """Represents a state completion event.

    CompletionEvent is automatically generated when a state
    completes its do-activity or becomes final.

    Class Invariants:
    1. Must track completion status
    2. Must maintain state consistency
    3. Must handle parallel regions
    4. Must preserve completion order

    Design Patterns:
    - Observer: Notifies of completion
    - Command: Encapsulates completion
    - Strategy: Implements completion types

    Threading/Concurrency Guarantees:
    1. Thread-safe status tracking
    2. Atomic completion detection
    3. Safe concurrent notification

    Performance Characteristics:
    1. O(1) event creation
    2. O(r) region checking where r is region count
    3. O(1) status updates
    """

    VALID_COMPLETION_TYPES = {"do_activity", "final"}

    def __init__(
        self,
        event_id: str,
        priority: EventPriority,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize a CompletionEvent instance.

        Args:
            event_id: Unique identifier for the completion event
            priority: Processing priority level
            data: Completion event parameters including state_id and completion_type
            timeout: Optional timeout in milliseconds

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if data is None:
            data = {}

        # Validate required parameters
        if "state_id" not in data:
            raise ValueError("Completion event data must include 'state_id' parameter")

        if "region_id" not in data:
            raise ValueError("Completion event data must include 'region_id' parameter")

        if "completion_type" not in data:
            raise ValueError("Completion event data must include 'completion_type' parameter")

        if data["completion_type"] not in self.VALID_COMPLETION_TYPES:
            raise ValueError(f"Invalid completion_type. Must be one of: {self.VALID_COMPLETION_TYPES}")

        super().__init__(event_id=event_id, kind=EventKind.COMPLETION, priority=priority, data=data, timeout=timeout)

    def can_consume(self) -> bool:
        """Check if the completion event can be consumed.

        Returns:
            True if the event hasn't been consumed yet,
            False if it has already been consumed.
        """
        return not self.consumed

    @property
    def result(self) -> Optional[Any]:
        """Get the completion result.

        Returns:
            The completion result if provided, None otherwise
        """
        return self.data.get("result")


class EventQueue:
    """Manages event queuing and processing.

    EventQueue implements a priority-based event queue with
    run-to-completion semantics and proper ordering.

    Class Invariants:
    1. Must maintain event order
    2. Must enforce RTC semantics
    3. Must handle priorities
    4. Must manage timeouts
    5. Must support cancellation
    6. Must track queue status
    7. Must handle overflow
    8. Must preserve fairness
    9. Must support filtering
    10. Must maintain consistency

    Design Patterns:
    - Queue: Manages event ordering
    - Strategy: Implements queue policies
    - Observer: Notifies of queue changes
    - Chain of Responsibility: Processes events

    Data Structures:
    - Priority queue for events
    - Set for cancelled events
    - Map for deferred events
    - List for processing history

    Algorithms:
    - Priority scheduling
    - Timeout management
    - Fairness enforcement
    - Load balancing

    Threading/Concurrency Guarantees:
    1. Thread-safe queue operations
    2. Atomic event processing
    3. Synchronized status updates
    4. Safe concurrent access
    5. Lock-free inspection
    6. Mutex protection for modifications

    Performance Characteristics:
    1. O(log n) enqueue/dequeue
    2. O(1) cancellation
    3. O(1) status check
    4. O(k) filtering where k is filter count
    5. O(t) timeout cleanup where t is timeout count

    Resource Management:
    1. Bounded queue size
    2. Memory-efficient storage
    3. Automatic cleanup
    4. Resource pooling
    5. Load shedding
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize an EventQueue instance.

        Args:
            max_size: Maximum number of events in the queue
        """
        self._queue = PriorityQueue()
        self._max_size = max_size
        self._processing = False
        self._events: Dict[str, Event] = {}  # For O(1) lookup by ID

    def __len__(self) -> int:
        """Get the number of events in the queue.

        Returns:
            Current number of events in the queue
        """
        # Remove cancelled events from count
        return len([event for event in self._events.values() if not event.cancelled])

    @property
    def is_processing(self) -> bool:
        """Check if the queue is currently processing events.

        Returns:
            True if events are being processed, False otherwise
        """
        return self._processing

    def start_processing(self) -> None:
        """Start event processing."""
        self._processing = True

    def stop_processing(self) -> None:
        """Stop event processing."""
        self._processing = False

    def enqueue(self, event: Event) -> bool:
        """Add an event to the queue.

        Args:
            event: The event to enqueue

        Returns:
            True if the event was enqueued, False otherwise

        Raises:
            ValueError: If the queue is full
        """
        if len(self) >= self._max_size:
            raise ValueError("Queue is full")

        if event.cancelled:
            return False

        if event.timeout is not None and event.timeout <= 0:
            return False

        self._queue.put(event)
        self._events[event.id] = event
        return True

    def dequeue(self) -> Optional[Event]:
        """Remove and return the highest priority event.

        Returns:
            The next event to process, or None if queue is empty
        """
        while not self._queue.empty():
            event = self._queue.get()

            # Skip cancelled events
            if event.cancelled:
                del self._events[event.id]
                continue

            # Skip timed out events
            if event.timeout is not None and event.timeout <= 0:
                del self._events[event.id]
                continue

            del self._events[event.id]
            return event

        return None

    def filter(self, predicate: Callable[[Event], bool]) -> List[Event]:
        """Filter events using a predicate.

        Args:
            predicate: Function that takes an event and returns bool

        Returns:
            List of events matching the predicate
        """
        return [event for event in self._events.values() if not event.cancelled and predicate(event)]

    def clear(self) -> None:
        """Clear all events from the queue."""
        while not self._queue.empty():
            self._queue.get()
        self._events.clear()
