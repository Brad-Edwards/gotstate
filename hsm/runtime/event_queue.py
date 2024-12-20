# hsm/runtime/event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hsm.core.events import Event


class EventQueue:
    """
    A simple event queue that provides events to the state machine executor.
    Can operate in either FIFO or priority mode.
    """

    def __init__(self, priority: bool = False) -> None:
        """
        Create a queue. If priority is True, use a priority-based structure.

        :param priority: Enable priority-based queueing.
        """
        raise NotImplementedError()

    def enqueue(self, event: "Event") -> None:
        """
        Add an event to the queue.

        :param event: The event to enqueue.
        """
        raise NotImplementedError()

    def dequeue(self) -> "Event" | None:
        """
        Remove and return the next event from the queue, or None if empty.

        :return: The next event or None if queue is empty.
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """
        Remove all events from the queue.
        """
        raise NotImplementedError()

    @property
    def priority_mode(self) -> bool:
        """
        Indicates whether this queue operates in priority mode.
        """
        raise NotImplementedError()


class _PriorityQueueWrapper:
    """
    Internal wrapper providing priority-based insertion and retrieval of events,
    if priority mode is enabled.
    """

    def __init__(self) -> None:
        """
        Initialize internal priority structure.
        """
        raise NotImplementedError()

    def push(self, event: "Event", priority: int) -> None:
        """
        Insert an event into the priority structure.
        """
        raise NotImplementedError()

    def pop(self) -> "Event" | None:
        """
        Retrieve and remove the highest priority event.
        """
        raise NotImplementedError()


class _EventQueueLock:
    """
    Internal context manager ensuring thread-safe access to the event queue.
    """

    def __init__(self) -> None:
        """
        Prepare internal locking mechanism.
        """
        raise NotImplementedError()

    def __enter__(self) -> None:
        """
        Acquire the lock when entering the context.
        """
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release the lock when exiting the context.
        """
        raise NotImplementedError()
