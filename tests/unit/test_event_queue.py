# tests/unit/test_event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest


def test_event_queue_fifo(mock_event):
    from hsm.runtime.event_queue import EventQueue

    eq = EventQueue(priority=False)
    eq.enqueue(mock_event)
    out = eq.dequeue()
    assert out == mock_event
    assert eq.dequeue() is None, "Queue should be empty now."


def test_event_queue_clear(mock_event):
    from hsm.runtime.event_queue import EventQueue

    eq = EventQueue(priority=False)
    eq.enqueue(mock_event)
    eq.clear()
    assert eq.dequeue() is None, "Clearing should remove all events."


def test_event_queue_priority(mock_event):
    from hsm.runtime.event_queue import EventQueue

    eq = EventQueue(priority=True)
    # Mock event objects with priority attributes if needed
    # We'll assume EventQueue sets priority internally. Since we have no direct API,
    # we just ensure it doesn't fail and returns events.
    eq.enqueue(mock_event)
    out = eq.dequeue()
    assert out == mock_event
    assert eq.priority_mode is True
