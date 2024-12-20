# hsm/runtime/timers.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import time
from typing import List

from hsm.core.events import TimeoutEvent


class Timer:
    """
    Represents a scheduled timeout. Useful for triggering TimeoutEvents when no
    other events occur within a certain timeframe.
    """

    def __init__(self, deadline: float) -> None:
        """
        Create a timer that expires at 'deadline' (timestamp or relative time).

        :param deadline: The point in time or offset when the timer expires.
        """
        self._deadline = deadline

    def is_expired(self) -> bool:
        """
        Check if the timer has reached its deadline.

        :return: True if expired, False otherwise.
        """
        return time.time() >= self._deadline

    @property
    def deadline(self) -> float:
        """
        Access the timer's configured expiration time.
        """
        return self._deadline


class _TimeoutRegistry:
    """
    Internal component maintaining a list of timers and associated events,
    allowing for expiration checks.
    """

    def __init__(self) -> None:
        """
        Prepare internal structures to track timeouts.
        """
        self._entries = []  # list of (Timer, TimeoutEvent) tuples

    def add(self, event: TimeoutEvent) -> None:
        """
        Record a TimeoutEvent for later checks.
        """
        t = Timer(event.deadline)
        self._entries.append((t, event))

    def expired_events(self) -> List[TimeoutEvent]:
        """
        Return a list of TimeoutEvents whose time has passed.
        Remove them from the registry.
        """
        now = time.time()
        expired = [(timer, evt) for (timer, evt) in self._entries if now >= timer.deadline]
        # Remove expired entries
        self._entries = [(t, e) for (t, e) in self._entries if (t, e) not in expired]
        return [evt for (t, evt) in expired]


class _TimeSource:
    """
    Abstract definition for obtaining the current time. Allows custom time sources
    (e.g., monotonic time) to be plugged in if needed.
    """

    def now(self) -> float:
        """
        Return the current time as a float (e.g., UNIX timestamp).
        """
        return time.time()


class TimeoutScheduler:
    """
    Schedules and manages TimeoutEvents by monitoring timers and firing events
    into the queue when deadlines pass.
    """

    def __init__(self) -> None:
        """
        Initialize the timeout scheduler.
        """
        self._registry = _TimeoutRegistry()

    def schedule_timeout(self, event: TimeoutEvent) -> None:
        """
        Add a TimeoutEvent to the schedule. The event will be triggered when its
        deadline is reached.

        :param event: The TimeoutEvent to schedule.
        """
        self._registry.add(event)

    def check_timeouts(self) -> List[TimeoutEvent]:
        """
        Check all scheduled timers, returning any that have expired and should be processed.

        :return: List of TimeoutEvents ready to be triggered.
        """
        return self._registry.expired_events()
