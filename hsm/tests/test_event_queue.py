# hsm/tests/test_event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import pytest

from hsm.core.events import Event
from hsm.runtime.event_queue import (
    AsyncEventQueue,
    EventQueue,
    EventQueueError,
    PrioritizedEvent,
    QueueEmptyError,
    QueueFullError,
    QueueStats,
)

# -----------------------------------------------------------------------------
# TEST FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def event_queue() -> EventQueue:
    """Fixture providing a standard event queue for testing."""
    return EventQueue(max_size=5)


@pytest.fixture
def async_event_queue() -> AsyncEventQueue:
    """Fixture providing an async event queue for testing."""
    return AsyncEventQueue(max_size=5)


@pytest.fixture
def test_event() -> Event:
    """Fixture providing a standard test event."""
    return Event("test_event", payload={"test": "data"}, priority=1)


@pytest.fixture
def queue_stats() -> QueueStats:
    """Fixture providing clean queue statistics."""
    return QueueStats()


# -----------------------------------------------------------------------------
# EXCEPTION TESTS
# -----------------------------------------------------------------------------


def test_queue_full_error() -> None:
    """Test QueueFullError creation and attributes."""
    error = QueueFullError("Queue is full", max_size=5, current_size=5)
    assert error.max_size == 5
    assert error.current_size == 5
    assert "Queue is full" in str(error)
    assert error.details == {"max_size": 5, "current_size": 5}


def test_queue_empty_error() -> None:
    """Test QueueEmptyError creation."""
    error = QueueEmptyError("Queue is empty")
    assert "Queue is empty" in str(error)


def test_event_queue_error() -> None:
    """Test EventQueueError creation with details."""
    details = {"operation": "enqueue", "queue_size": 5}
    error = EventQueueError("Operation failed", details)
    assert error.details == details
    assert "Operation failed" in str(error)


# -----------------------------------------------------------------------------
# QUEUE STATS TESTS
# -----------------------------------------------------------------------------


def test_queue_stats_recording(queue_stats: QueueStats) -> None:
    """Test QueueStats recording operations."""
    queue_stats.record_enqueue(5)
    assert queue_stats.total_enqueued == 1
    assert queue_stats.max_seen_size == 5

    queue_stats.record_dequeue(0.1)
    assert queue_stats.total_dequeued == 1
    assert queue_stats.total_wait_time == pytest.approx(0.1)


def test_queue_stats_max_seen_size(queue_stats: QueueStats) -> None:
    """Test QueueStats tracks maximum size correctly."""
    queue_stats.record_enqueue(3)
    queue_stats.record_enqueue(5)
    queue_stats.record_enqueue(2)
    assert queue_stats.max_seen_size == 5


# -----------------------------------------------------------------------------
# PRIORITIZED EVENT TESTS
# -----------------------------------------------------------------------------


def test_prioritized_event_ordering() -> None:
    """Test PrioritizedEvent comparison and ordering."""
    event1 = Event("event1", priority=1)
    event2 = Event("event2", priority=2)

    p_event1 = PrioritizedEvent(priority=1, sequence=1, event=event1)
    p_event2 = PrioritizedEvent(priority=2, sequence=2, event=event2)
    p_event3 = PrioritizedEvent(priority=1, sequence=3, event=event1)

    assert p_event1 < p_event2  # Lower priority comes first
    assert p_event1 < p_event3  # Same priority, earlier sequence comes first
    assert p_event1 != p_event2


def test_prioritized_event_equality() -> None:
    """Test PrioritizedEvent equality comparison."""
    event = Event("test")
    p_event1 = PrioritizedEvent(priority=1, sequence=1, event=event)
    p_event2 = PrioritizedEvent(priority=1, sequence=1, event=event)
    p_event3 = PrioritizedEvent(priority=1, sequence=2, event=event)

    assert p_event1 == p_event2
    assert p_event1 != p_event3
    assert p_event1 != "not_an_event"


# -----------------------------------------------------------------------------
# SYNCHRONOUS QUEUE TESTS
# -----------------------------------------------------------------------------


def test_event_queue_initialization() -> None:
    """Test EventQueue initialization with various parameters."""
    queue = EventQueue(max_size=5)
    assert queue.size() == 0
    assert not queue.is_full()
    assert queue.is_empty()

    with pytest.raises(ValueError):
        EventQueue(max_size=0)

    with pytest.raises(ValueError):
        EventQueue(max_size=-1)


def test_event_queue_enqueue_dequeue(event_queue: EventQueue, test_event: Event) -> None:
    """Test basic enqueue and dequeue operations."""
    event_queue.enqueue(test_event)
    assert not event_queue.is_empty()
    assert event_queue.size() == 1

    dequeued = event_queue.dequeue()
    assert dequeued.get_id() == test_event.get_id()
    assert event_queue.is_empty()


def test_event_queue_priority_ordering(event_queue: EventQueue) -> None:
    """Test that events are dequeued in priority order."""
    event1 = Event("event1", priority=2)
    event2 = Event("event2", priority=1)
    event3 = Event("event3", priority=3)

    event_queue.enqueue(event1)
    event_queue.enqueue(event2)
    event_queue.enqueue(event3)

    assert event_queue.dequeue().get_id() == "event2"  # Priority 1
    assert event_queue.dequeue().get_id() == "event1"  # Priority 2
    assert event_queue.dequeue().get_id() == "event3"  # Priority 3


def test_event_queue_full_error(event_queue: EventQueue) -> None:
    """Test queue full error handling."""
    for i in range(5):
        event_queue.enqueue(Event(f"event{i}"))

    with pytest.raises(QueueFullError) as exc_info:
        event_queue.enqueue(Event("overflow"))
    assert exc_info.value.max_size == 5
    assert exc_info.value.current_size == 5


def test_event_queue_empty_error(event_queue: EventQueue) -> None:
    """Test queue empty error handling."""
    with pytest.raises(QueueEmptyError):
        event_queue.dequeue()


def test_event_queue_shutdown(event_queue: EventQueue, test_event: Event) -> None:
    """Test queue shutdown behavior."""
    event_queue.enqueue(test_event)
    event_queue.shutdown()

    with pytest.raises(EventQueueError):
        event_queue.enqueue(test_event)

    with pytest.raises(EventQueueError):
        event_queue.dequeue()


def test_event_queue_clear(event_queue: EventQueue) -> None:
    """Test queue clear operation."""
    for i in range(3):
        event_queue.enqueue(Event(f"event{i}"))

    event_queue.clear()
    assert event_queue.is_empty()
    assert event_queue.size() == 0


def test_event_queue_peek(event_queue: EventQueue, test_event: Event) -> None:
    """Test queue peek operation."""
    event_queue.enqueue(test_event)

    peeked = event_queue.peek()
    assert peeked is not None
    assert peeked.get_id() == test_event.get_id()
    assert event_queue.size() == 1  # Peek shouldn't remove the event


def test_event_queue_try_dequeue(event_queue: EventQueue, test_event: Event) -> None:
    """Test try_dequeue operation with timeout."""
    assert event_queue.try_dequeue(timeout=0.1) is None

    event_queue.enqueue(test_event)
    dequeued = event_queue.try_dequeue(timeout=0.1)
    assert dequeued is not None
    assert dequeued.get_id() == test_event.get_id()


def test_event_queue_atomic_operations(event_queue: EventQueue) -> None:
    """Test that queue operations maintain atomicity."""
    events = [Event(f"event{i}", priority=i) for i in range(3)]

    # Test atomic enqueue
    for event in events:
        event_queue.enqueue(event)

    # Verify queue state
    assert event_queue.size() == 3
    assert not event_queue.is_full()

    # Test atomic dequeue
    dequeued_events = []
    while not event_queue.is_empty():
        dequeued_events.append(event_queue.dequeue())

    # Verify ordering
    assert len(dequeued_events) == 3
    assert [e.get_priority() for e in dequeued_events] == [0, 1, 2]


# -----------------------------------------------------------------------------
# ASYNCHRONOUS QUEUE TESTS
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_event_queue_initialization() -> None:
    """Test AsyncEventQueue initialization."""
    queue = AsyncEventQueue(max_size=5)
    assert await queue.size() == 0
    assert await queue.is_empty()

    with pytest.raises(ValueError):
        AsyncEventQueue(max_size=0)


@pytest.mark.asyncio
async def test_async_event_queue_enqueue_dequeue(async_event_queue: AsyncEventQueue, test_event: Event) -> None:
    """Test async enqueue and dequeue operations."""
    await async_event_queue.enqueue(test_event)
    assert not await async_event_queue.is_empty()
    assert await async_event_queue.size() == 1

    dequeued = await async_event_queue.dequeue()
    assert dequeued.get_id() == test_event.get_id()
    assert await async_event_queue.is_empty()


@pytest.mark.asyncio
async def test_async_event_queue_priority_ordering(async_event_queue: AsyncEventQueue) -> None:
    """Test async queue priority ordering."""
    event1 = Event("event1", priority=2)
    event2 = Event("event2", priority=1)
    event3 = Event("event3", priority=3)

    await async_event_queue.enqueue(event1)
    await async_event_queue.enqueue(event2)
    await async_event_queue.enqueue(event3)

    assert (await async_event_queue.dequeue()).get_id() == "event2"  # Priority 1
    assert (await async_event_queue.dequeue()).get_id() == "event1"  # Priority 2
    assert (await async_event_queue.dequeue()).get_id() == "event3"  # Priority 3


@pytest.mark.asyncio
async def test_async_event_queue_full_error(async_event_queue: AsyncEventQueue) -> None:
    """Test async queue full error handling."""
    for i in range(5):
        await async_event_queue.enqueue(Event(f"event{i}"))

    with pytest.raises(QueueFullError) as exc_info:
        await async_event_queue.enqueue(Event("overflow"))
    assert exc_info.value.max_size == 5
    assert exc_info.value.current_size == 5


@pytest.mark.asyncio
async def test_async_event_queue_empty_error(async_event_queue: AsyncEventQueue) -> None:
    """Test async queue empty error handling."""
    with pytest.raises(QueueEmptyError):
        await async_event_queue.dequeue()


@pytest.mark.asyncio
async def test_async_event_queue_shutdown(async_event_queue: AsyncEventQueue, test_event: Event) -> None:
    """Test async queue shutdown behavior."""
    await async_event_queue.enqueue(test_event)
    await async_event_queue.shutdown()

    with pytest.raises(EventQueueError):
        await async_event_queue.enqueue(test_event)

    with pytest.raises(EventQueueError):
        await async_event_queue.dequeue()


@pytest.mark.asyncio
async def test_async_event_queue_try_dequeue(async_event_queue: AsyncEventQueue, test_event: Event) -> None:
    """Test async try_dequeue operation."""
    assert await async_event_queue.try_dequeue(timeout=0.1) is None

    await async_event_queue.enqueue(test_event)
    dequeued = await async_event_queue.try_dequeue(timeout=0.1)
    assert dequeued is not None
    assert dequeued.get_id() == test_event.get_id()


# -----------------------------------------------------------------------------
# CONCURRENCY TESTS
# -----------------------------------------------------------------------------


def test_event_queue_thread_safety(event_queue: EventQueue) -> None:
    """Test thread safety of EventQueue operations."""

    def producer() -> None:
        for i in range(50):
            try:
                event_queue.enqueue(Event(f"event{i}", priority=i % 3))
                time.sleep(0.001)
            except QueueFullError:
                pass

    def consumer() -> None:
        for _ in range(50):
            try:
                event_queue.dequeue()
                time.sleep(0.001)
            except QueueEmptyError:
                pass

    threads = [
        threading.Thread(target=producer),
        threading.Thread(target=producer),
        threading.Thread(target=consumer),
        threading.Thread(target=consumer),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Queue should be in a consistent state
    assert event_queue.size() >= 0
    assert event_queue.size() <= event_queue._max_size


@pytest.mark.asyncio
async def test_async_event_queue_concurrency(async_event_queue: AsyncEventQueue) -> None:
    """Test concurrency of AsyncEventQueue operations."""

    async def producer() -> None:
        for i in range(50):
            try:
                await async_event_queue.enqueue(Event(f"event{i}", priority=i % 3))
                await asyncio.sleep(0.001)
            except QueueFullError:
                pass

    async def consumer() -> None:
        for _ in range(50):
            try:
                await async_event_queue.dequeue()
                await asyncio.sleep(0.001)
            except QueueEmptyError:
                pass

    tasks = [
        asyncio.create_task(producer()),
        asyncio.create_task(producer()),
        asyncio.create_task(consumer()),
        asyncio.create_task(consumer()),
    ]

    await asyncio.gather(*tasks)

    # Queue should be in a consistent state
    assert await async_event_queue.size() >= 0
    assert await async_event_queue.size() <= async_event_queue._max_size


# -----------------------------------------------------------------------------
# LOGGING TESTS
# -----------------------------------------------------------------------------


def test_event_queue_logging(caplog: Any, event_queue: EventQueue, test_event: Event) -> None:
    """Test that queue operations are properly logged."""
    caplog.set_level(logging.DEBUG)

    event_queue.enqueue(test_event)
    assert "Enqueued event" in caplog.text
    assert test_event.get_id() in caplog.text

    event_queue.dequeue()
    assert "Dequeued event" in caplog.text
    assert "wait_time" in caplog.text


@pytest.mark.asyncio
async def test_async_event_queue_logging(caplog: Any, async_event_queue: AsyncEventQueue, test_event: Event) -> None:
    """Test that async queue operations are properly logged."""
    caplog.set_level(logging.DEBUG)

    await async_event_queue.enqueue(test_event)
    assert "Enqueued event" in caplog.text
    assert test_event.get_id() in caplog.text

    await async_event_queue.dequeue()
    assert "Dequeued event" in caplog.text
    assert "wait_time" in caplog.text
