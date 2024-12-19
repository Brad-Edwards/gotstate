# hsm/tests/test_event_queue.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import gc
import inspect
import logging
import random
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, Union

import psutil
import pytest
from async_timeout import timeout
from hypothesis import given
from hypothesis import strategies as st

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


# -----------------------------------------------------------------------------
# BOUNDARY AND EDGE CASES
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_exact_max_size(queue_class, is_async):
    """Test queue behavior when reaching exactly max_size."""
    queue = queue_class(max_size=2)
    event1 = Event("event1")
    event2 = Event("event2")

    # Fill queue to exact max_size
    await queue_op(queue, "enqueue", event1, is_async=is_async)
    await queue_op(queue, "enqueue", event2, is_async=is_async)
    assert await queue_op(queue, "size", is_async=is_async) == 2

    # Verify queue is full
    with pytest.raises(QueueFullError):
        await queue_op(queue, "enqueue", Event("event3"), is_async=is_async)

    # Remove one event and verify we can add again
    await queue_op(queue, "dequeue", is_async=is_async)
    await queue_op(queue, "enqueue", Event("event3"), is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_priority_edge_cases(queue_class, is_async):
    """Test edge cases in priority handling."""
    queue = queue_class(max_size=5)

    # Test equal priorities
    event1 = Event("event1", priority=1)
    event2 = Event("event2", priority=1)
    event3 = Event("event3", priority=1)

    await queue_op(queue, "enqueue", event1, is_async=is_async)
    await queue_op(queue, "enqueue", event2, is_async=is_async)
    await queue_op(queue, "enqueue", event3, is_async=is_async)

    # Should maintain FIFO order for equal priorities
    dequeued1 = await queue_op(queue, "dequeue", is_async=is_async)
    dequeued2 = await queue_op(queue, "dequeue", is_async=is_async)
    dequeued3 = await queue_op(queue, "dequeue", is_async=is_async)

    assert dequeued1.get_id() == "event1"
    assert dequeued2.get_id() == "event2"
    assert dequeued3.get_id() == "event3"

    # Test negative priorities
    neg_event = Event("negative", priority=-10)
    pos_event = Event("positive", priority=10)
    zero_event = Event("zero", priority=0)

    await queue_op(queue, "enqueue", pos_event, is_async=is_async)
    await queue_op(queue, "enqueue", zero_event, is_async=is_async)
    await queue_op(queue, "enqueue", neg_event, is_async=is_async)

    # Negative priority should come first
    assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == "negative"
    assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == "zero"
    assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == "positive"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_state_transitions(queue_class, is_async):
    """Test queue behavior during state transitions (empty↔full)."""
    queue = queue_class(max_size=2)
    event = Event("test")

    # Empty → Full
    assert await queue_op(queue, "is_empty", is_async=is_async)
    await queue_op(queue, "enqueue", event, is_async=is_async)
    assert not await queue_op(queue, "is_empty", is_async=is_async)
    assert not await queue_op(queue, "is_full", is_async=is_async)
    await queue_op(queue, "enqueue", event, is_async=is_async)
    assert await queue_op(queue, "is_full", is_async=is_async)

    # Full → Empty
    await queue_op(queue, "dequeue", is_async=is_async)
    assert not await queue_op(queue, "is_full", is_async=is_async)
    await queue_op(queue, "dequeue", is_async=is_async)
    assert await queue_op(queue, "is_empty", is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_timeout_boundary_values(queue_class, is_async):
    """Test boundary values for timeout parameter."""
    queue = queue_class(max_size=1)

    # Test zero timeout
    result = await queue_op(queue, "try_dequeue", timeout=0, is_async=is_async)
    assert result is None

    # Test very small timeout
    result = await queue_op(queue, "try_dequeue", timeout=1e-6, is_async=is_async)
    assert result is None

    # Test negative timeout (should be treated as zero)
    result = await queue_op(queue, "try_dequeue", timeout=-1, is_async=is_async)
    assert result is None


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_stats_edge_cases(queue_class, is_async):
    """Test edge cases for queue statistics."""
    queue = queue_class(max_size=2)
    stats = await queue_op(queue, "get_stats", is_async=is_async)

    # Test stats for rapid enqueue/dequeue
    event = Event("test")
    for _ in range(100):
        await queue_op(queue, "enqueue", event, is_async=is_async)
        await queue_op(queue, "dequeue", is_async=is_async)

    stats = await queue_op(queue, "get_stats", is_async=is_async)
    assert stats.total_enqueued == 100
    assert stats.total_dequeued == 100
    assert stats.max_seen_size == 1

    # Test stats after clear
    await queue_op(queue, "clear", is_async=is_async)
    stats = await queue_op(queue, "get_stats", is_async=is_async)
    assert stats.total_enqueued == 100  # Should not reset
    assert stats.total_dequeued == 100  # Should not reset


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_event_timestamp_handling(queue_class, is_async):
    """Test handling of event timestamps during queue operations."""
    queue = queue_class(max_size=3)

    # Create events with different creation times
    events = [Event("event1"), Event("event2"), Event("event3")]

    # Enqueue in reverse order
    for event in reversed(events):
        await queue_op(queue, "enqueue", event, is_async=is_async)

    # Verify FIFO order is maintained
    for expected_event in reversed(events):  # Should dequeue in LIFO order
        dequeued = await queue_op(queue, "dequeue", is_async=is_async)
        assert dequeued.get_id() == expected_event.get_id()


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_peek_operations(queue_class, is_async):
    """Test peek operations in various queue states."""
    queue = queue_class(max_size=2)

    # Peek empty queue
    assert await queue_op(queue, "peek", is_async=is_async) is None

    # Peek with one event
    event1 = Event("event1", priority=1)
    await queue_op(queue, "enqueue", event1, is_async=is_async)
    peeked = await queue_op(queue, "peek", is_async=is_async)
    assert peeked.get_id() == "event1"
    assert await queue_op(queue, "size", is_async=is_async) == 1  # Verify peek didn't remove

    # Peek with multiple events
    event2 = Event("event2", priority=0)  # Higher priority
    await queue_op(queue, "enqueue", event2, is_async=is_async)
    peeked = await queue_op(queue, "peek", is_async=is_async)
    assert peeked.get_id() == "event2"  # Should see highest priority

    # Verify peek after dequeue
    await queue_op(queue, "dequeue", is_async=is_async)
    peeked = await queue_op(queue, "peek", is_async=is_async)
    assert peeked.get_id() == "event1"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_sequence_numbers(queue_class, is_async):
    """Test sequence number handling for equal priority events."""
    queue = queue_class(max_size=3)

    # Enqueue events with equal priority but track sequence
    events = [Event(f"event{i}", priority=1) for i in range(3)]
    for event in events:
        await queue_op(queue, "enqueue", event, is_async=is_async)

    # Verify FIFO ordering is maintained by sequence numbers
    for i in range(3):
        dequeued = await queue_op(queue, "dequeue", is_async=is_async)
        assert dequeued.get_id() == f"event{i}"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_invalid_event_handling(queue_class, is_async):
    """Test handling of invalid event types."""
    queue = queue_class(max_size=1)

    # Test with None event
    with pytest.raises(TypeError):
        await queue_op(queue, "enqueue", None, is_async=is_async)

    # Test with non-Event object
    class FakeEvent:
        pass

    with pytest.raises(TypeError):
        await queue_op(queue, "enqueue", FakeEvent(), is_async=is_async)

    # Test with invalid event implementation
    class InvalidEvent:
        def get_id(self) -> str:
            return "invalid"

        # Missing get_priority method

    with pytest.raises(TypeError):
        await queue_op(queue, "enqueue", InvalidEvent(), is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_detailed_statistics(queue_class, is_async):
    """Test detailed queue statistics tracking."""
    queue = queue_class(max_size=2)

    # Test initial stats
    stats = await queue_op(queue, "get_stats", is_async=is_async)
    assert stats.total_enqueued == 0
    assert stats.total_dequeued == 0
    assert stats.max_seen_size == 0
    assert stats.total_wait_time == pytest.approx(0.0, abs=1e-6)
    assert stats.last_operation_time >= 0.0


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_operation_timestamps(queue_class, is_async):
    """Test operation timestamp tracking in statistics."""
    queue = queue_class(max_size=1)

    # Record initial timestamp
    initial_time = time.monotonic()
    await queue_op(queue, "enqueue", Event("test"), is_async=is_async)
    stats1 = await queue_op(queue, "get_stats", is_async=is_async)
    enqueue_time = stats1.last_operation_time

    # Wait and perform another operation
    await asyncio.sleep(0.1)

    # Get timestamp before dequeue
    before_dequeue = time.monotonic()
    await queue_op(queue, "dequeue", is_async=is_async)
    stats2 = await queue_op(queue, "get_stats", is_async=is_async)
    dequeue_time = stats2.last_operation_time

    # Verify timestamps are monotonically increasing
    assert enqueue_time >= initial_time, "First operation time should be after initial time"
    assert dequeue_time >= before_dequeue, "Dequeue time should be after before_dequeue time"
    assert dequeue_time > enqueue_time, "Dequeue time should be after enqueue time"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
async def test_extreme_priority_values(queue_class, is_async):
    """Test handling of extreme priority values."""
    queue = queue_class(max_size=5)
    events = [Event("max", priority=sys.maxsize), Event("min", priority=-sys.maxsize - 1), Event("zero", priority=0)]
    for event in events:
        await queue_op(queue, "enqueue", event, is_async=is_async)

    # Verify correct ordering
    assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == "min"
    assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == "zero"
    assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == "max"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
async def test_sequence_number_wraparound(queue_class, is_async):
    """Test sequence number handling at boundaries."""
    queue = queue_class(max_size=3)
    queue._sequence = sys.maxsize  # Force sequence to boundary

    events = [Event(f"event{i}") for i in range(3)]
    for event in events:
        await queue_op(queue, "enqueue", event, is_async=is_async)

    # Verify FIFO still maintained after wraparound
    for i in range(3):
        assert (await queue_op(queue, "dequeue", is_async=is_async)).get_id() == f"event{i}"


def test_concurrent_enqueue_dequeue():
    """Test thread safety of concurrent operations."""
    queue = EventQueue(max_size=100)
    event_count = 1000
    producer_count = 4
    consumer_count = 4

    produced = []
    consumed = []
    stop_consumers = threading.Event()
    producer_done = threading.Event()

    def producer():
        success_count = 0
        while success_count < event_count:
            try:
                event = Event(f"event{success_count}")
                queue.enqueue(event)
                produced.append(event.get_id())
                success_count += 1
            except QueueFullError:
                time.sleep(0.001)

    def consumer():
        while not (stop_consumers.is_set() and queue.is_empty()):
            try:
                event = queue.try_dequeue(timeout=0.1)
                if event:
                    consumed.append(event.get_id())
            except QueueEmptyError:
                if producer_done.is_set() and queue.is_empty():
                    break
                time.sleep(0.001)

    # Start producers
    producer_threads = []
    for _ in range(producer_count):
        t = threading.Thread(target=producer)
        producer_threads.append(t)
        t.start()

    # Start consumers
    consumer_threads = []
    for _ in range(consumer_count):
        t = threading.Thread(target=consumer)
        consumer_threads.append(t)
        t.start()

    # Wait for producers with timeout
    for t in producer_threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "Producer thread timed out"

    producer_done.set()  # Signal producers are done

    # Wait a bit more for consumers to finish
    time.sleep(0.5)
    stop_consumers.set()  # Signal consumers to stop

    # Wait for consumers with timeout
    for t in consumer_threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "Consumer thread timed out"

    # Verify all events were processed
    expected_total = event_count * producer_count
    assert len(produced) == expected_total, f"Expected {expected_total} produced events, got {len(produced)}"
    assert len(consumed) == expected_total, f"Expected {expected_total} consumed events, got {len(consumed)}"
    assert sorted(produced) == sorted(consumed), "Produced and consumed events don't match"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_resource_cleanup(queue_class, is_async):
    """Test proper resource cleanup during shutdown."""
    queue = queue_class(max_size=5)

    # Fill queue
    for i in range(5):
        await queue_op(queue, "enqueue", Event(f"event{i}"), is_async=is_async)

    # Start operation that will block
    if is_async:
        enqueue_task = asyncio.create_task(queue.enqueue(Event("blocked")))
    else:

        def enqueue_with_handling():
            try:
                queue.enqueue(Event("blocked"))
            except (QueueFullError, EventQueueError):
                pass  # Expected exceptions during shutdown

        enqueue_thread = threading.Thread(target=enqueue_with_handling)
        enqueue_thread.start()

    await asyncio.sleep(0.1)  # Let operation block

    # Shutdown should unblock operations
    await queue_op(queue, "shutdown", is_async=is_async)

    if is_async:
        with pytest.raises(EventQueueError):
            await enqueue_task
    else:
        enqueue_thread.join(timeout=1.0)
        assert not enqueue_thread.is_alive()


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_error_recovery(queue_class, is_async):
    """Test queue recovery after errors."""
    queue = queue_class(max_size=2)

    # Test recovery after full queue
    for i in range(2):
        await queue_op(queue, "enqueue", Event(f"event{i}"), is_async=is_async)

    with pytest.raises(QueueFullError):
        await queue_op(queue, "enqueue", Event("overflow"), is_async=is_async)

    # Should still be able to dequeue
    event = await queue_op(queue, "dequeue", is_async=is_async)
    assert event.get_id() == "event0"

    # Should be able to enqueue again
    await queue_op(queue, "enqueue", Event("new"), is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_clear_operations(queue_class, is_async):
    """Test clear operation behavior."""
    queue = queue_class(max_size=3)

    # Test clear on empty queue
    await queue_op(queue, "clear", is_async=is_async)
    assert await queue_op(queue, "is_empty", is_async=is_async)

    # Test clear with items
    for i in range(3):
        await queue_op(queue, "enqueue", Event(f"event{i}"), is_async=is_async)
    await queue_op(queue, "clear", is_async=is_async)
    assert await queue_op(queue, "is_empty", is_async=is_async)

    # Verify can still use queue after clear
    await queue_op(queue, "enqueue", Event("new"), is_async=is_async)
    assert await queue_op(queue, "size", is_async=is_async) == 1


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_peek_operations(queue_class, is_async):
    """Test peek operation behavior."""
    queue = queue_class(max_size=2)

    # Peek empty queue
    assert await queue_op(queue, "peek", is_async=is_async) is None

    # Peek with items
    event1 = Event("event1", priority=2)
    event2 = Event("event2", priority=1)
    await queue_op(queue, "enqueue", event1, is_async=is_async)
    await queue_op(queue, "enqueue", event2, is_async=is_async)

    # Should see highest priority without removing
    peeked = await queue_op(queue, "peek", is_async=is_async)
    assert peeked.get_id() == "event2"
    assert await queue_op(queue, "size", is_async=is_async) == 2

    # Peek after dequeue
    await queue_op(queue, "dequeue", is_async=is_async)
    peeked = await queue_op(queue, "peek", is_async=is_async)
    assert peeked.get_id() == "event1"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_event_validation(queue_class, is_async):
    """Test event validation."""
    queue = queue_class(max_size=1)

    # Test None event
    with pytest.raises(TypeError):
        await queue_op(queue, "enqueue", None, is_async=is_async)

    # Test with non-Event object
    class FakeEvent:
        pass

    with pytest.raises(TypeError):
        await queue_op(queue, "enqueue", FakeEvent(), is_async=is_async)

    # Test with invalid Event implementation
    class InvalidEvent:
        def get_id(self) -> str:
            return "invalid"

        # Missing get_priority method

    with pytest.raises(TypeError):
        await queue_op(queue, "enqueue", InvalidEvent(), is_async=is_async)


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_timeout_edge_cases(queue_class, is_async):
    """Test timeout behavior edge cases."""
    queue = queue_class(max_size=1)

    # Test zero timeout
    result = await queue_op(queue, "try_dequeue", timeout=0, is_async=is_async)
    assert result is None

    # Test negative timeout
    result = await queue_op(queue, "try_dequeue", timeout=-1, is_async=is_async)
    assert result is None

    # Test very small timeout
    result = await queue_op(queue, "try_dequeue", timeout=1e-6, is_async=is_async)
    assert result is None

    # Test timeout with data
    await queue_op(queue, "enqueue", Event("test"), is_async=is_async)
    result = await queue_op(queue, "try_dequeue", timeout=0.1, is_async=is_async)
    assert result is not None
    assert result.get_id() == "test"


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_logging_verification(queue_class, is_async, caplog):
    """Test logging behavior."""
    queue = queue_class(max_size=2)
    caplog.set_level(logging.DEBUG)

    # Test enqueue logging
    event = Event("test_event", priority=1)
    await queue_op(queue, "enqueue", event, is_async=is_async)
    assert "Enqueued event test_event" in caplog.text
    assert "priority=1" in caplog.text

    # Test dequeue logging
    await queue_op(queue, "dequeue", is_async=is_async)
    assert "Dequeued event test_event" in caplog.text
    assert "wait_time=" in caplog.text


@pytest.mark.stress
@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_stress(queue_class, is_async):
    """Test queue behavior under heavy load."""
    queue = queue_class(max_size=1000)
    event_count = 10000
    batch_size = 500

    # Generate lots of events with random priorities
    events = [Event(f"event{i}", priority=random.randint(-100, 100)) for i in range(event_count)]

    # Process events in batches
    dequeued = []
    for i in range(0, event_count, batch_size):
        batch = events[i : i + batch_size]

        # Sort just this batch to get expected order
        batch_expected = sorted(
            [(e.get_priority(), i, e.get_id()) for i, e in enumerate(batch)], key=lambda x: (x[0], x[1])
        )

        # Enqueue batch
        for event in batch:
            await queue_op(queue, "enqueue", event, is_async=is_async)

        # Dequeue batch and verify order
        batch_dequeued = []
        while not await queue_op(queue, "is_empty", is_async=is_async):
            event = await queue_op(queue, "dequeue", is_async=is_async)
            batch_dequeued.append((event.get_priority(), event.get_id()))

        # Compare just this batch's order
        for (exp_prio, _, exp_id), (act_prio, act_id) in zip(batch_expected, batch_dequeued):
            if exp_prio != act_prio:
                print(f"\nFirst ordering violation in batch {i//batch_size}:")
                print(f"Expected: priority={exp_prio}, id={exp_id}")
                print(f"Actual: priority={act_prio}, id={act_id}")
                assert exp_prio == act_prio, "Priority ordering violated"

        dequeued.extend(batch_dequeued)

    # Final verification
    assert len(dequeued) == len(events), f"Expected {len(events)} events, got {len(dequeued)}"


@pytest.mark.property
@given(
    events=st.lists(
        st.tuples(st.text(min_size=1, max_size=10), st.integers(min_value=-100, max_value=100)),  # event ID  # priority
        min_size=0,
        max_size=100,
    )
)
@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_properties(events, queue_class, is_async):
    """Property-based tests for queue behavior."""
    queue = queue_class(max_size=None)  # Unlimited queue for property testing

    # Convert generated data to events
    events = [Event(id_, priority=prio) for id_, prio in events]

    # Track expected order
    expected_order = sorted([(e.get_priority(), i, e.get_id()) for i, e in enumerate(events)])

    # Enqueue all events
    for event in events:
        await queue_op(queue, "enqueue", event, is_async=is_async)

    # Dequeue and verify properties
    dequeued = []
    while not await queue_op(queue, "is_empty", is_async=is_async):
        event = await queue_op(queue, "dequeue", is_async=is_async)
        dequeued.append((event.get_priority(), event.get_id()))

    # Property 1: Length preservation
    assert len(dequeued) == len(events), "Queue should preserve event count"

    # Property 2: Priority ordering
    if dequeued:
        assert all(a[0] <= b[0] for a, b in zip(dequeued, dequeued[1:])), "Priority ordering violated"


@pytest.mark.memory
@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_memory_pressure(queue_class, is_async):
    """Test queue behavior under memory pressure."""
    queue = queue_class(max_size=1000)
    event_refs = []

    # Monitor initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Create and enqueue events while tracking references
    for i in range(1000):
        event = Event(f"event{i}", payload={"data": "x" * 1000})  # 1KB payload
        event_refs.append(weakref.ref(event))
        await queue_op(queue, "enqueue", event, is_async=is_async)
        event = None  # Clear the reference in this scope

    # Clear all events
    await queue_op(queue, "clear", is_async=is_async)

    # Clear any local references that might be holding onto events
    queue = None

    # Force garbage collection multiple times to ensure cleanup
    for _ in range(3):
        gc.collect()

    # Debug: Print surviving references
    surviving_refs = [ref for ref in event_refs if ref() is not None]
    if surviving_refs:
        print(f"\nSurviving event references: {len(surviving_refs)}")
        print(f"First surviving event: {surviving_refs[0]()}")
        # Get referrers to see what's holding the reference
        import gc as garbage_collector  # Use a different name to avoid shadowing

        referrers = garbage_collector.get_referrers(surviving_refs[0]())
        print("Objects referring to surviving event:")
        for ref in referrers:
            print(f"  {type(ref)}: {ref}")
            # If it's a frame, print more details
            if isinstance(ref, type(garbage_collector.current_frame())):
                print(f"    Frame locals: {ref.f_locals}")

    # Move the assertion into a separate function to avoid frame reference
    def check_remaining_refs():
        remaining = sum(1 for ref in event_refs if ref() is not None)
        assert remaining == 0, f"{remaining} events were not properly cleaned up"

    check_remaining_refs()


@pytest.mark.stress
@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_rapid_enqueue_dequeue(queue_class, is_async):
    """Test queue behavior with rapid enqueue/dequeue operations."""
    queue = queue_class(max_size=100)
    operation_count = 10000

    async def rapid_operations():
        for i in range(operation_count):
            try:
                if i % 2 == 0:
                    await queue_op(queue, "enqueue", Event(f"event{i}"), is_async=is_async)
                else:
                    await queue_op(queue, "try_dequeue", timeout=0.001, is_async=is_async)
            except (QueueFullError, QueueEmptyError):
                continue

    # Run multiple operation coroutines concurrently
    tasks = [rapid_operations() for _ in range(4)]
    await asyncio.gather(*tasks)

    # Verify queue is in valid state
    size = await queue_op(queue, "size", is_async=is_async)
    assert 0 <= size <= 100, f"Queue size {size} outside valid range"


@pytest.mark.stress
@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_shutdown_under_load(queue_class, is_async):
    """Test queue shutdown behavior under load."""
    queue = queue_class(max_size=1000)
    shutdown_flag = asyncio.Event()
    tasks = []

    async def load_generator():
        try:
            while not shutdown_flag.is_set():
                try:
                    # Use shorter timeouts for operations
                    if random.random() < 0.5:
                        await asyncio.wait_for(
                            queue_op(queue, "enqueue", Event("test"), is_async=is_async), timeout=0.1
                        )
                    else:
                        await asyncio.wait_for(
                            queue_op(queue, "try_dequeue", timeout=0.001, is_async=is_async), timeout=0.1
                        )
                except (QueueFullError, QueueEmptyError, EventQueueError, asyncio.TimeoutError, asyncio.CancelledError):
                    if shutdown_flag.is_set():
                        return
                    await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            return

    async with timeout(5.0):
        try:
            # Start load generators
            tasks = [asyncio.create_task(load_generator()) for _ in range(4)]

            # Let them run briefly
            await asyncio.sleep(0.1)

            # Signal shutdown
            shutdown_flag.set()

            # Shutdown queue
            try:
                await asyncio.wait_for(queue_op(queue, "shutdown", is_async=is_async), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # Continue to cleanup even if shutdown times out

            # Cancel all tasks immediately
            for task in tasks:
                task.cancel()

            # Wait for tasks with short timeout
            done, pending = await asyncio.wait(tasks, timeout=1.0)

            # Force cancel any remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await asyncio.wait_for(task, 0.1)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Verify shutdown state
            with pytest.raises(EventQueueError):
                await queue_op(queue, "enqueue", Event("test"), is_async=is_async)

        finally:
            # Aggressive cleanup
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, 0.1)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass  # Ignore any errors during cleanup


@pytest.mark.parametrize("queue_class,is_async", [(EventQueue, False), (AsyncEventQueue, True)])
@pytest.mark.asyncio
async def test_queue_priority_simple(queue_class, is_async):
    """Test simple priority ordering with controlled input."""
    queue = queue_class(max_size=10)

    # Create events with known priorities
    events = [
        Event("event1", priority=-100),
        Event("event2", priority=-100),
        Event("event3", priority=-99),
        Event("event4", priority=-99),
        Event("event5", priority=-100),
    ]

    # Enqueue all events
    for event in events:
        await queue_op(queue, "enqueue", event, is_async=is_async)

    # Dequeue and verify order
    dequeued = []
    while not await queue_op(queue, "is_empty", is_async=is_async):
        event = await queue_op(queue, "dequeue", is_async=is_async)
        dequeued.append((event.get_priority(), event.get_id()))
        print(f"Dequeued: priority={event.get_priority()}, id={event.get_id()}")

    # Verify all -100 priority events come before -99 priority events
    priorities = [p for p, _ in dequeued]
    for i in range(len(priorities) - 1):
        assert (
            priorities[i] <= priorities[i + 1]
        ), f"Priority ordering violated at index {i}: {priorities[i]} > {priorities[i+1]}"
