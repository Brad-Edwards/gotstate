# hsm/runtime/event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import heapq
import threading
from collections import deque
from typing import Optional

from hsm.core.events import Event


class _EventQueueLock:
    """
    Internal context manager ensuring thread-safe access to the event queue.
    """

    def __init__(self, lock: threading.Lock) -> None:
        """
        Prepare internal locking mechanism.
        """
        self._lock = lock

    def __enter__(self) -> None:
        """
        Acquire the lock when entering the context.
        """
        self._lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release the lock when exiting the context.
        """
        self._lock.release()


class _PriorityQueueWrapper:
    """
    Internal wrapper providing priority-based insertion and retrieval of events,
    if priority mode is enabled. Events with higher priority values are processed first.
    Within the same priority level, events are processed in FIFO order.
    """

    def __init__(self) -> None:
        """
        Initialize internal priority structure.
        """
        self._counter = 0
        self._heap = []

    def push(self, event: Event) -> None:
        """
        Insert an event into the priority structure. Higher priority events will be
        processed first. For events with equal priority, FIFO order is maintained.

        :param event: The event to enqueue.
        """
        # Negative priority for max-heap behavior (higher priority first)
        # Counter ensures FIFO order within same priority
        heapq.heappush(self._heap, (-event.priority, self._counter, event))
        self._counter += 1

    def pop(self) -> Optional[Event]:
        """
        Retrieve and remove the next event in priority order.
        Higher priority events are returned first, with FIFO ordering within priority levels.

        :return: The next event or None if queue is empty.
        """
        if not self._heap:
            return None
        _, _, event = heapq.heappop(self._heap)
        return event

    def clear(self) -> None:
        """
        Clear all events.
        """
        self._heap.clear()
        self._counter = 0


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
        self._priority_mode = priority
        self._lock = threading.Lock()

        if self._priority_mode:
            self._queue = _PriorityQueueWrapper()
        else:
            self._queue = deque()

    def enqueue(self, event: Event) -> None:
        """
        Add an event to the queue.

        :param event: The event to enqueue.
        """
        with _EventQueueLock(self._lock):
            if self._priority_mode:
                self._queue.push(event)
            else:
                self._queue.append(event)

    def dequeue(self) -> Optional[Event]:
        """
        Remove and return the next event from the queue, or None if empty.

        :return: The next event or None if queue is empty.
        """
        with _EventQueueLock(self._lock):
            if self._priority_mode:
                return self._queue.pop()
            else:
                if self._queue:
                    return self._queue.popleft()
                return None

    def clear(self) -> None:
        """
        Remove all events from the queue.
        """
        with _EventQueueLock(self._lock):
            self._queue.clear()

    @property
    def priority_mode(self) -> bool:
        """
        Indicates whether this queue operates in priority mode.
        """
        return self._priority_mode
