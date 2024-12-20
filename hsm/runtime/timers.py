# hsm/runtime/timers.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    def is_expired(self) -> bool:
        """
        Check if the timer has reached its deadline.

        :return: True if expired, False otherwise.
        """

    @property
    def deadline(self) -> float:
        """
        Access the timer's configured expiration time.
        """


class TimeoutScheduler:
    """
    Schedules and manages TimeoutEvents by monitoring timers and firing events
    into the queue when deadlines pass.
    """

    def __init__(self) -> None:
        """
        Initialize the timeout scheduler.
        """

    def schedule_timeout(self, event: "TimeoutEvent") -> None:
        """
        Add a TimeoutEvent to the schedule. The event will be triggered when its
        deadline is reached.

        :param event: The TimeoutEvent to schedule.
        """

    def check_timeouts(self) -> list["TimeoutEvent"]:
        """
        Check all scheduled timers, returning any that have expired and should be processed.

        :return: List of TimeoutEvents ready to be triggered.
        """


class _TimeoutRegistry:
    """
    Internal component maintaining a list of timers and associated events,
    allowing for expiration checks.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def add(self, event: "TimeoutEvent") -> None:
        raise NotImplementedError()

    def expired_events(self) -> list["TimeoutEvent"]:
        raise NotImplementedError()


class _TimeSource:
    """
    Abstract definition for obtaining the current time. Allows custom time sources
    (e.g., monotonic time) to be plugged in if needed.
    """

    def now(self) -> float:
        raise NotImplementedError()
