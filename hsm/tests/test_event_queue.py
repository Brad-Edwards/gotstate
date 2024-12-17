# hsm/tests/test_event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import inspect
import logging
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Union

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
def queue_stats() -> QueueStats:
    """Fixture providing clean queue statistics."""
    return QueueStats()


@pytest.fixture
def queue_types():
    """Fixture providing both queue types for parametrized tests."""
    return [(EventQueue, False), (AsyncEventQueue, True)]


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
# PARAMETRIZED TESTS
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_initialization(queue_class, is_async):
    """Test queue initialization for both sync and async queues."""
    queue = queue_class(max_size=5)

    size = await async_wrapper(queue.size) if is_async else queue.size()
    is_empty = await async_wrapper(queue.is_empty) if is_async else queue.is_empty()

    assert size == 0
    assert is_empty

    with pytest.raises(ValueError):
        queue_class(max_size=0)

    with pytest.raises(ValueError):
        queue_class(max_size=-1)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_basic_operations(queue_class, is_async):
    """Test basic operations for both sync and async queues."""
    queue = queue_class(max_size=5)
    test_event = Event("test_event", payload={"test": "data"}, priority=1)

    # Test enqueue
    if is_async:
        await queue.enqueue(test_event)
        assert not await queue.is_empty()
        assert await queue.size() == 1
    else:
        queue.enqueue(test_event)
        assert not queue.is_empty()
        assert queue.size() == 1

    # Test dequeue
    dequeued = await async_wrapper(queue.dequeue) if is_async else queue.dequeue()
    assert dequeued.get_id() == test_event.get_id()
    assert await async_wrapper(queue.is_empty) if is_async else queue.is_empty()


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_priority_ordering(queue_class, is_async):
    """Test priority ordering for both sync and async queues."""
    queue = queue_class(max_size=5)
    events = [Event("event1", priority=2), Event("event2", priority=1), Event("event3", priority=3)]

    for event in events:
        if is_async:
            await queue.enqueue(event)
        else:
            queue.enqueue(event)

    # Verify priority ordering
    expected_order = ["event2", "event1", "event3"]
    for expected_id in expected_order:
        dequeued = await async_wrapper(queue.dequeue) if is_async else queue.dequeue()
        assert dequeued.get_id() == expected_id


async def async_wrapper(func: Callable, *args, **kwargs):
    """Helper to handle both async and sync functions."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


async def queue_op(queue, operation: str, *args, is_async=False, **kwargs):
    """Helper for queue operations that handles both sync and async cases."""
    func = getattr(queue, operation)
    if is_async:
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_error_conditions(queue_class, is_async):
    """Test queue error conditions for both sync and async queues."""
    queue = queue_class(max_size=2)
    test_event = Event("test_event")

    # Test queue full error
    await queue_op(queue, "enqueue", test_event, is_async=is_async)
    await queue_op(queue, "enqueue", test_event, is_async=is_async)

    with pytest.raises(QueueFullError) as exc_info:
        await queue_op(queue, "enqueue", test_event, is_async=is_async)
    assert exc_info.value.max_size == 2
    assert exc_info.value.current_size == 2

    # Test queue empty error
    queue = queue_class(max_size=2)  # Fresh queue
    with pytest.raises(QueueEmptyError):
        await queue_op(queue, "dequeue", is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_shutdown_behavior(queue_class, is_async):
    """Test shutdown behavior for both sync and async queues."""
    queue = queue_class(max_size=2)
    test_event = Event("test_event")

    await queue_op(queue, "enqueue", test_event, is_async=is_async)
    await queue_op(queue, "shutdown", is_async=is_async)

    with pytest.raises(EventQueueError):
        await queue_op(queue, "enqueue", test_event, is_async=is_async)

    with pytest.raises(EventQueueError):
        await queue_op(queue, "dequeue", is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_try_dequeue(queue_class, is_async):
    """Test try_dequeue behavior for both sync and async queues."""
    queue = queue_class(max_size=2)
    test_event = Event("test_event")

    # Try dequeue on empty queue
    result = await queue_op(queue, "try_dequeue", timeout=0.1, is_async=is_async)
    assert result is None

    # Try dequeue with event
    await queue_op(queue, "enqueue", test_event, is_async=is_async)
    dequeued = await queue_op(queue, "try_dequeue", timeout=0.1, is_async=is_async)
    assert dequeued is not None
    assert dequeued.get_id() == test_event.get_id()


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_logging(queue_class, is_async, caplog: Any):
    """Test logging behavior for both sync and async queues."""
    queue = queue_class(max_size=2)
    test_event = Event("test_event")
    caplog.set_level(logging.DEBUG)

    await queue_op(queue, "enqueue", test_event, is_async=is_async)
    assert "Enqueued event" in caplog.text
    assert test_event.get_id() in caplog.text

    await queue_op(queue, "dequeue", is_async=is_async)
    assert "Dequeued event" in caplog.text
    assert "wait_time" in caplog.text


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_async_queue_concurrent_operations(queue_class, is_async):
    """Test concurrent operations for async queue."""
    if not is_async:
        pytest.skip("This test is for async queues only")

    queue = queue_class(max_size=5)
    tasks = await create_producer_consumer_tasks(queue)
    await asyncio.gather(*tasks)

    # Queue should be in a consistent state
    assert await queue.size() >= 0
    assert await queue.size() <= queue._max_size


async def create_producer_consumer_tasks(queue):
    """Create producer and consumer tasks for async queue testing."""
    return [
        asyncio.create_task(async_producer(queue)),
        asyncio.create_task(async_producer(queue)),
        asyncio.create_task(async_consumer(queue)),
        asyncio.create_task(async_consumer(queue)),
    ]


async def async_producer(queue) -> None:
    """Async producer function for queue testing."""
    for i in range(50):
        try:
            await queue.enqueue(Event(f"event{i}", priority=i % 3))
            await asyncio.sleep(0.001)
        except QueueFullError:
            pass


async def async_consumer(queue) -> None:
    """Async consumer function for queue testing."""
    for _ in range(50):
        try:
            await queue.dequeue()
            await asyncio.sleep(0.001)
        except QueueEmptyError:
            pass


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
def test_sync_queue_concurrent_operations(queue_class, is_async):
    """Test concurrent operations for sync queue."""
    if is_async:
        pytest.skip("This test is for sync queues only")

    queue = queue_class(max_size=5)
    threads = create_producer_consumer_threads(queue)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Queue should be in a consistent state
    assert queue.size() >= 0
    assert queue.size() <= queue._max_size


def create_producer_consumer_threads(queue):
    """Create producer and consumer threads for sync queue testing."""
    return [
        threading.Thread(target=sync_producer, args=(queue,)),
        threading.Thread(target=sync_producer, args=(queue,)),
        threading.Thread(target=sync_consumer, args=(queue,)),
        threading.Thread(target=sync_consumer, args=(queue,)),
    ]


def sync_producer(queue) -> None:
    """Sync producer function for queue testing."""
    for i in range(50):
        try:
            queue.enqueue(Event(f"event{i}", priority=i % 3))
            time.sleep(0.001)
        except QueueFullError:
            pass


def sync_consumer(queue) -> None:
    """Sync consumer function for queue testing."""
    for _ in range(50):
        try:
            queue.dequeue()
            time.sleep(0.001)
        except QueueEmptyError:
            pass


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_concurrent_stress(queue_class, is_async):
    """Test queue behavior under concurrent stress with rapid enqueue/dequeue."""
    queue = queue_class(max_size=5)
    stats = {"operations_completed": 0}

    if is_async:
        await run_async_stress_test(queue, stats)
    else:
        run_sync_stress_test(queue, stats)

    # Verify the test did meaningful work
    assert stats["operations_completed"] > 0
    # Verify queue remains in valid state
    size = await queue.size() if is_async else queue.size()
    assert 0 <= size <= queue._max_size


async def run_async_stress_test(queue, stats):
    """Run stress test with async workers."""
    workers = [asyncio.create_task(stress_worker(queue, stats, is_async=True)) for _ in range(4)]
    await asyncio.gather(*workers)


def run_sync_stress_test(queue, stats):
    """Run stress test with sync workers."""
    threads = [
        threading.Thread(target=lambda: asyncio.run(stress_worker(queue, stats, is_async=False))) for _ in range(4)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()


async def stress_worker(queue, stats: dict, is_async: bool) -> None:
    """Worker function for stress testing the queue."""
    for _ in range(20):
        try:
            if random.random() < 0.5:  # NOSONAR
                await handle_enqueue_operation(queue, is_async)
            else:
                await handle_dequeue_operation(queue, is_async)
            stats["operations_completed"] += 1
            await asyncio.sleep(0.001) if is_async else time.sleep(0.001)
        except (QueueFullError, QueueEmptyError):
            continue


async def handle_enqueue_operation(queue, is_async: bool):
    """Handle enqueue operation for stress testing."""
    event = Event("stress_test", priority=random.randint(1, 3))  # NOSONAR
    if is_async:
        await queue.enqueue(event)
    else:
        queue.enqueue(event)


async def handle_dequeue_operation(queue, is_async: bool):
    """Handle dequeue operation for stress testing."""
    if is_async:
        await queue.try_dequeue(timeout=0.001)
    else:
        queue.try_dequeue(timeout=0.001)
