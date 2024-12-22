# hsm/core/events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import time
from typing import Any, Dict


class Event:
    """
    Represents a signal or trigger within the state machine. Events cause the
    machine to evaluate transitions and possibly change states.
    """

    def __init__(self, name: str, priority: int = 0) -> None:
        """
        Create an event identified by a name. Metadata may be attached as needed.

        :param name: A string identifying this event.
        :param priority: Priority level for this event (higher numbers = higher priority).
        """
        self._name = name
        self._priority = priority
        self._metadata: Dict[str, Any] = {}
        self._timestamp = time.time()

    @property
    def name(self) -> str:
        """The name of the event."""
        return self._name

    @property
    def priority(self) -> int:
        """The priority level of the event."""
        return self._priority

    @property
    def metadata(self) -> Dict[str, Any]:
        """Optional dictionary of additional event data."""
        return self._metadata

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        # Higher priority numbers come first
        if self.priority != other.priority:
            return self.priority > other.priority
        # For same priority, use timestamp (FIFO)
        return self._timestamp < other._timestamp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return (
            isinstance(other, Event)
            and self.name == other.name
            and self.priority == other.priority
            and self._timestamp == other._timestamp
        )


class TimeoutEvent(Event):
    """
    A special event that fires after a timeout. Often used to force state machine
    transitions if an expected event does not arrive within a given timeframe.
    """

    def __init__(self, name: str, deadline: float) -> None:
        """
        Initialize a timeout event with a deadline.

        :param name: Event name.
        :param deadline: A timestamp or duration indicating when to fire.
        """
        super().__init__(name)
        self._deadline = deadline

    @property
    def deadline(self) -> float:
        """The time at which this event should be triggered if no other events occur."""
        return self._deadline
