"""
Time and change event scheduling management.

Manages time events, change events, timer lifecycle,
and event coordination.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

import icontract

from gotstate.exceptions import GotStateError

logger = logging.getLogger(__name__)


class TimerStatus(Enum):
    """Defines the possible states of a timer."""

    IDLE = auto()
    ACTIVE = auto()
    PAUSED = auto()
    CANCELLED = auto()
    EXPIRED = auto()


class TimerKind(Enum):
    """Defines the different types of timers."""

    RELATIVE = auto()
    ABSOLUTE = auto()
    PERIODIC = auto()


@icontract.invariant(lambda self: isinstance(self._timers, dict), "Timers collection must be a dict")
class Scheduler:
    """Manages time and change event scheduling.

    Provides centralized timer and change event management with
    thread-safe operations.

    Class Invariants:
    1. Timers collection must be a dict
    """

    def __init__(self) -> None:
        self._timers: Dict[str, threading.Timer] = {}
        self._timer_statuses: Dict[str, TimerStatus] = {}
        self._change_conditions: Dict[str, Callable[[], bool]] = {}
        self._lock = threading.RLock()

    @icontract.require(
        lambda timer_id: isinstance(timer_id, str) and len(timer_id) > 0, "Timer ID must be a non-empty string"
    )
    @icontract.require(lambda duration: duration > 0, "Duration must be positive")
    @icontract.require(lambda callback: callable(callback), "Callback must be callable")
    def schedule_timer(self, timer_id: str, duration: float, callback: Callable[[], None]) -> None:
        """Schedule a relative timer."""
        with self._lock:
            if timer_id in self._timers:
                raise GotStateError(f"Timer '{timer_id}' already exists")
            timer = threading.Timer(duration, self._on_timer_expired, args=[timer_id, callback])
            timer.daemon = True
            self._timers[timer_id] = timer
            self._timer_statuses[timer_id] = TimerStatus.ACTIVE
            timer.start()

    def _on_timer_expired(self, timer_id: str, callback: Callable[[], None]) -> None:
        with self._lock:
            self._timer_statuses[timer_id] = TimerStatus.EXPIRED
        try:
            callback()
        except Exception:
            logger.exception("Timer callback failed for timer '%s'", timer_id)

    def cancel_timer(self, timer_id: str) -> None:
        """Cancel a scheduled timer."""
        with self._lock:
            if timer_id not in self._timers:
                raise GotStateError(f"Timer '{timer_id}' not found")
            self._timers[timer_id].cancel()
            self._timer_statuses[timer_id] = TimerStatus.CANCELLED

    def get_timer_status(self, timer_id: str) -> TimerStatus:
        with self._lock:
            if timer_id not in self._timer_statuses:
                raise GotStateError(f"Timer '{timer_id}' not found")
            return self._timer_statuses[timer_id]

    @icontract.require(
        lambda condition_id: isinstance(condition_id, str) and len(condition_id) > 0,
        "Condition ID must be a non-empty string",
    )
    @icontract.require(lambda condition: callable(condition), "Condition must be callable")
    def register_change_condition(self, condition_id: str, condition: Callable[[], bool]) -> None:
        """Register a condition to monitor for changes."""
        with self._lock:
            self._change_conditions[condition_id] = condition

    def check_conditions(self) -> List[str]:
        """Check all registered conditions and return IDs of those that are True."""
        triggered: List[str] = []
        with self._lock:
            for cid, condition in self._change_conditions.items():
                try:
                    if condition():
                        triggered.append(cid)
                except Exception:
                    pass
        return triggered

    def cancel_all(self) -> None:
        """Cancel all active timers."""
        with self._lock:
            for timer_id, timer in self._timers.items():
                timer.cancel()
                self._timer_statuses[timer_id] = TimerStatus.CANCELLED
