# tests/unit/test_event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from threading import Lock
from unittest.mock import MagicMock

import pytest

from gotstate.core.events import Event
from gotstate.runtime.event_queue import EventQueue


def test_event_queue_fifo(mock_event):
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=False)
    eq.enqueue(mock_event)
    out = eq.dequeue()
    assert out == mock_event
    assert eq.dequeue() is None, "Queue should be empty now."


def test_event_queue_clear(mock_event):
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=False)
    eq.enqueue(mock_event)
    eq.clear()
    assert eq.dequeue() is None, "Clearing should remove all events."


def test_event_queue_priority(mock_event):
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=True)
    # Mock event objects with priority attributes if needed
    # We'll assume EventQueue sets priority internally. Since we have no direct API,
    # we just ensure it doesn't fail and returns events.
    eq.enqueue(mock_event)
    out = eq.dequeue()
    assert out == mock_event
    assert eq.priority_mode is True


def test_event_queue_lock():
    from gotstate.runtime.event_queue import _EventQueueLock

    mock_lock = MagicMock()
    mock_lock.acquire = MagicMock()
    mock_lock.release = MagicMock()

    with _EventQueueLock(mock_lock):
        mock_lock.acquire.assert_called_once()
    mock_lock.release.assert_called_once()


def test_event_queue_lock_exception():
    from gotstate.runtime.event_queue import _EventQueueLock

    mock_lock = MagicMock()
    mock_lock.acquire = MagicMock()
    mock_lock.release = MagicMock()

    try:
        with _EventQueueLock(mock_lock):
            raise Exception("Test error")
    except Exception:
        pass
    mock_lock.acquire.assert_called_once()
    mock_lock.release.assert_called_once()


def test_priority_queue_wrapper():
    from gotstate.core.events import Event
    from gotstate.runtime.event_queue import _PriorityQueueWrapper

    pq = _PriorityQueueWrapper()
    event1 = Event("Event1")
    event2 = Event("Event2")

    pq.push(event1)
    pq.push(event2)

    # Events should come out in FIFO order since no priorities are set
    assert pq.pop() == event1
    assert pq.pop() == event2
    assert pq.pop() is None


def test_priority_queue_wrapper_clear():
    from gotstate.core.events import Event
    from gotstate.runtime.event_queue import _PriorityQueueWrapper

    pq = _PriorityQueueWrapper()
    pq.push(Event("Event1"))
    pq.clear()
    assert pq.pop() is None
    assert pq._counter == 0


def test_event_queue_fifo_empty():
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=False)
    assert eq.dequeue() is None


def test_event_queue_priority_empty():
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=True)
    assert eq.dequeue() is None


def test_event_queue_multiple_events(mock_event):
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=False)
    eq.enqueue(mock_event)
    eq.enqueue(mock_event)

    assert eq.dequeue() == mock_event
    assert eq.dequeue() == mock_event
    assert eq.dequeue() is None


def test_event_queue_priority_multiple_events(mock_event):
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue(priority=True)
    eq.enqueue(mock_event)
    eq.enqueue(mock_event)

    assert eq.dequeue() == mock_event
    assert eq.dequeue() == mock_event
    assert eq.dequeue() is None


def test_event_queue_clear_empty():
    from gotstate.runtime.event_queue import EventQueue

    eq = EventQueue()
    eq.clear()  # Should not raise any errors
    assert eq.dequeue() is None


def test_priority_queue_ordering():
    """Test that events are properly ordered by priority"""
    queue = EventQueue(priority=True)

    queue.enqueue(Event("low", priority=1))
    queue.enqueue(Event("high", priority=10))
    queue.enqueue(Event("medium", priority=5))

    assert queue.dequeue().name == "high"
    assert queue.dequeue().name == "medium"
    assert queue.dequeue().name == "low"
