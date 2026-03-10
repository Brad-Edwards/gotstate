"""
State machine monitoring and metrics management.

Provides introspection capabilities, event emission,
and metric collection for state machine execution.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional

import icontract


class MonitoringLevel(Enum):
    """Defines monitoring detail levels."""

    MINIMAL = auto()
    NORMAL = auto()
    DETAILED = auto()
    DEBUG = auto()


class MetricType(Enum):
    """Defines types of metrics to collect."""

    STATE = auto()
    TRANSITION = auto()
    EVENT = auto()
    RESOURCE = auto()
    ERROR = auto()


@icontract.invariant(lambda self: isinstance(self._level, MonitoringLevel), "Monitoring level must be valid")
class Monitor:
    """Provides state machine monitoring and metrics.

    Tracks state machine behavior and collects performance metrics
    with configurable detail levels.

    Class Invariants:
    1. Monitoring level must be a valid MonitoringLevel
    """

    def __init__(self, level: MonitoringLevel = MonitoringLevel.NORMAL, history_size: int = 1000) -> None:
        self._level = level
        self._metrics: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()

    @property
    def level(self) -> MonitoringLevel:
        return self._level

    @level.setter
    def level(self, value: MonitoringLevel) -> None:
        self._level = value

    @property
    def metrics(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._metrics)

    @property
    def counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        with self._lock:
            self._metrics[name] = value

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit a monitoring event to all subscribers."""
        record = {
            "type": event_type,
            "timestamp": time.monotonic(),
            "data": data or {},
        }
        with self._lock:
            self._history.append(record)
        for subscriber in self._subscribers:
            try:
                subscriber(record)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to monitoring events."""
        self._subscribers.append(callback)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return recent monitoring events."""
        with self._lock:
            if limit is not None:
                return list(self._history)[-limit:]
            return list(self._history)

    def reset(self) -> None:
        """Reset all metrics and history."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._history.clear()
