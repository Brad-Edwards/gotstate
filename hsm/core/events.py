# hsm/core/events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
class Event:
    """
    Represents a signal or trigger within the state machine. Events cause the
    machine to evaluate transitions and possibly change states.
    """

    def __init__(self, name: str) -> None:
        """
        Create an event identified by a name. Metadata may be attached later.

        :param name: A string identifying this event.
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        The name of the event.
        """
        raise NotImplementedError()

    @property
    def metadata(self) -> dict[str, any]:
        """
        Optional dictionary of additional event data.
        """
        raise NotImplementedError()


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
        raise NotImplementedError()

    @property
    def deadline(self) -> float:
        """
        The time at which this event should be triggered if no other events occur.
        """
        raise NotImplementedError()


class _EventMetadataNormalizer:
    """
    Internal tool to standardize and clean event metadata dictionaries, ensuring
    consistent formatting and keys.
    """

    def normalize(self, metadata: dict[str, any]) -> dict[str, any]:
        """
        Normalize event metadata, returning a clean, consistent dictionary.
        """
        raise NotImplementedError()
