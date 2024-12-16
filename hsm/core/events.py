# hsm/core/events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
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

    Example:
        evt = Event("start", payload={"key": "value"}, priority=10)
        evt_id = evt.get_id()         # "start"
        data = evt.get_payload()      # {"key": "value"}
        prio = evt.get_priority()     # 10
    """

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0) -> None:
        if not event_id:
            raise ValueError("event_id cannot be empty")
        self._id = event_id
        self._payload = payload
        self._priority = priority

    def get_id(self) -> EventID:
        """Return the unique event identifier."""
        return self._id

    def get_payload(self) -> Any:
        """Return the event payload, which can be None or any object."""
        return self._payload

    def get_priority(self) -> int:
        """Return the event priority as an integer."""
        return self._priority


class TimeoutEvent(Event):
    """
    An event with an associated timeout value.

    Runtime Invariants:
    - timeout >= 0.0
    - Inherits all invariants from Event.

    Raises:
        ValueError: If timeout < 0.

    Example:
        tev = TimeoutEvent("timeout_event", payload={"action": "stop"}, priority=5, timeout=1.5)
        tev_id = tev.get_id()       # "timeout_event"
        tev_payload = tev.get_payload()  # {"action": "stop"}
        tev_prio = tev.get_priority()     # 5
        tev_timeout = tev.get_timeout()   # 1.5
    """

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0, timeout: float = 0.0) -> None:
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        super().__init__(event_id, payload, priority)
        self._timeout = timeout

    def get_timeout(self) -> float:
        """Return the timeout value associated with this event."""
        return self._timeout
