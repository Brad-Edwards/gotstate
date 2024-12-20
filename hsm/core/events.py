# hsm/core/events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import Any, Dict


class Event:
    """
    Represents a signal or trigger within the state machine. Events cause the
    machine to evaluate transitions and possibly change states.
    """

    def __init__(self, name: str) -> None:
        """
        Create an event identified by a name. Metadata may be attached as needed.

        :param name: A string identifying this event.
        """
        self._name = name
        self._metadata: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """The name of the event."""
        return self._name

    @property
    def metadata(self) -> Dict[str, Any]:
        """Optional dictionary of additional event data."""
        return self._metadata


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
