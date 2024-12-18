# hsm/core/events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import copy
from typing import Any

from hsm.interfaces.abc import AbstractEvent
from hsm.interfaces.types import EventID


class Event(AbstractEvent):
    """
    A basic event object implementing the AbstractEvent protocol.

    Runtime Invariants:
    - Events are immutable after creation.
    - event_id is a non-empty string.
    - priority is an integer.
    - payload can be any object or None.

    Raises:
        ValueError: If event_id is empty.
        TypeError: If event_id is not a string or priority is not an integer.

    Example:
        evt = Event("start", payload={"key": "value"}, priority=10)
        evt_id = evt.get_id()         # "start"
        data = evt.get_payload()      # {"key": "value"}
        prio = evt.get_priority()     # 10
    """

    def _validate_event_id(self, event_id: Any) -> None:
        """Validate event_id is a non-empty string."""
        if not isinstance(event_id, str):
            raise TypeError("event_id must be a string")
        if not event_id:
            raise ValueError("event_id cannot be empty")

    def _validate_priority(self, priority: Any) -> None:
        """Validate priority is an integer."""
        if not isinstance(priority, int):
            raise TypeError("priority must be an integer")

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0) -> None:
        self._validate_event_id(event_id)
        self._validate_priority(priority)

        self._id = event_id
        self._payload = copy.deepcopy(payload) if payload is not None else None
        self._priority = priority

    def _get_attributes(self) -> dict:
        """Get common attributes for string representation."""
        return {"id": self._id, "payload": self._payload, "priority": self._priority}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        if isinstance(self, TimeoutEvent) != isinstance(other, TimeoutEvent):
            return False  # One is TimeoutEvent, other is regular Event
        return self._id == other._id and self._payload == other._payload and self._priority == other._priority

    def __str__(self) -> str:
        attrs = self._get_attributes()
        return f"Event({', '.join(f'{k}={v}' for k, v in attrs.items())})"

    def get_id(self) -> EventID:
        """Return the unique event identifier."""
        return self._id

    def get_payload(self) -> Any:
        """Return the event payload, which can be None or any object."""
        return self._payload

    def get_priority(self) -> int:
        """Return the event priority as an integer."""
        return self._priority

    def __hash__(self) -> int:
        """Make Event instances hashable for use as dictionary keys."""
        return hash((self._id, str(self._payload), self._priority))


class TimeoutEvent(Event):
    """
    An event with an associated timeout value.

    Runtime Invariants:
    - timeout >= 0.0
    - Inherits all invariants from Event.

    Raises:
        ValueError: If timeout < 0.
        TypeError: If timeout is not a number.

    Example:
        tev = TimeoutEvent("timeout_event", payload={"action": "stop"}, priority=5, timeout=1.5)
        tev_id = tev.get_id()       # "timeout_event"
        tev_payload = tev.get_payload()  # {"action": "stop"}
        tev_prio = tev.get_priority()     # 5
        tev_timeout = tev.get_timeout()   # 1.5
    """

    def _validate_timeout(self, timeout: Any) -> None:
        """Validate timeout is a non-negative number."""
        if not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")
        if timeout < 0:
            raise ValueError("timeout cannot be negative")

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0, timeout: float = 0.0) -> None:
        self._validate_timeout(timeout)
        super().__init__(event_id, payload, priority)
        self._timeout = 0.0 if timeout == 0.0 else float(timeout)

    def _get_attributes(self) -> dict:
        """Get attributes including timeout for string representation."""
        attrs = super()._get_attributes()
        attrs["timeout"] = self._timeout
        return attrs

    def get_timeout(self) -> float:
        """Return the timeout value associated with this event."""
        return self._timeout

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeoutEvent):
            return NotImplemented
        return super().__eq__(other) and self._timeout == other._timeout

    def __hash__(self) -> int:
        """Make TimeoutEvent instances hashable."""
        return hash((super().__hash__(), self._timeout))

    def __str__(self) -> str:
        attrs = self._get_attributes()
        return f"TimeoutEvent({', '.join(f'{k}={v}' for k, v in attrs.items())})"
