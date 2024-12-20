"""Unit tests for async event queue."""

import asyncio
import pytest
from typing import List

from hsm.core.events import Event
from hsm.runtime.async_support import AsyncEventQueue

@pytest.fixture
async def queue():
    """Create a fresh async event queue for each test."""
    return AsyncEventQueue()

@pytest.mark.asyncio
async def test_basic_queue_operations(queue):
    """Test basic enqueue/dequeue operations."""
    event = Event("test")
    await queue.enqueue(event)
    dequeued = await queue.dequeue()
    assert dequeued == event
    
    # Queue should be empty after dequeue
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.dequeue(), timeout=0.1)

@pytest.mark.asyncio
async def test_priority_ordering(queue):
    """Test that events are processed in priority order."""
    events = [
        Event("low", priority=0),
        Event("high", priority=2),
        Event("medium", priority=1)
    ]
    
    # Enqueue events in random order
    for event in events:
        await queue.enqueue(event)
    
    # Should dequeue in priority order
    high = await queue.dequeue()
    medium = await queue.dequeue()
    low = await queue.dequeue()
    
    assert high.name == "high"
    assert medium.name == "medium"
    assert low.name == "low"

@pytest.mark.asyncio
async def test_queue_clear(queue):
    """Test clearing the queue."""
    # Fill queue with events
    events = [Event(f"event_{i}") for i in range(5)]
    for event in events:
        await queue.enqueue(event)
    
    # Clear queue
    await queue.clear()
    
    # Queue should be empty
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.dequeue(), timeout=0.1)

@pytest.mark.asyncio
async def test_queue_empty_timeout(queue):
    """Test timeout behavior when queue is empty."""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.dequeue(), timeout=0.1)

@pytest.mark.asyncio
async def test_multiple_consumers_single_producer(queue):
    """Test multiple consumers with a single producer."""
    event_count = 10
    events = [Event(f"event_{i}") for i in range(event_count)]
    received_events = []
    
    async def producer():
        for event in events:
            await queue.enqueue(event)
    
    async def consumer():
        try:
            while True:
                event = await asyncio.wait_for(queue.dequeue(), timeout=0.1)
                if event:
                    received_events.append(event)
        except asyncio.TimeoutError:
            pass  # Expected when queue is empty
    
    # Run producer and consumers
    await producer()
    consumers = [consumer() for _ in range(3)]
    await asyncio.gather(*consumers)
    
    # Verify all events were received
    assert len(received_events) == event_count
    assert {e.name for e in received_events} == {e.name for e in events}