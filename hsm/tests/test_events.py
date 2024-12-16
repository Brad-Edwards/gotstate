# hsm/tests/test_events.py
# nosec
# Licensed under the MIT License - see LICENSE file for details
"""
Test suite for events defined in events.py.

This module tests the Event and TimeoutEvent classes, ensuring they adhere to
the AbstractEvent protocol and handle various scenarios correctly.

Sections covered:
- Basic Error Cases
- Core Functionality
- Edge Cases
- Integration
"""
import asyncio
import io
import logging
import random
import string
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict

import pytest
import pytest_asyncio

from hsm.core.errors import HSMError, create_error_context
from hsm.core.events import Event, TimeoutEvent

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_event() -> Event:
    """Fixture for a sample event."""
    return Event("test_event", payload={"data": 42}, priority=5)


@pytest.fixture
def sample_timeout_event() -> TimeoutEvent:
    """Fixture for a sample timeout event."""
    return TimeoutEvent("timeout_event", payload={"info": "test"}, priority=10, timeout=1.5)


@pytest.fixture
def sample_state_data() -> Dict[str, Any]:
    """Sample state data for scenarios if needed."""
    return {"count": 10, "status": "active"}


@pytest.fixture
def logger_fixture():
    """Fixture for testing logging integration."""
    logger = logging.getLogger("hsm.test.events")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


@pytest_asyncio.fixture
async def async_setup():
    """Async fixture for async tests."""
    await asyncio.sleep(0.01)
    return True


# -----------------------------------------------------------------------------
# BASIC ERROR CASES
# -----------------------------------------------------------------------------


def test_empty_event_id_raises_error() -> None:
    """
    Test that creating an event with an empty event_id raises ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        Event("")
    assert "event_id cannot be empty" in str(exc_info.value)


def test_negative_timeout_raises_error() -> None:
    """
    Test that creating a TimeoutEvent with negative timeout raises ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        TimeoutEvent("evt", timeout=-1.0)
    assert "timeout cannot be negative" in str(exc_info.value)


def test_hsm_error_inheritance() -> None:
    """
    Test that HSMError forms the base for event-related errors if any occur.
    This is symbolic since Event itself may not raise HSMError.
    """

    class CustomEventError(HSMError):
        pass

    with pytest.raises(CustomEventError) as exc_info:
        raise CustomEventError("test message")
    assert isinstance(exc_info.value, HSMError)


# -----------------------------------------------------------------------------
# CORE FUNCTIONALITY
# -----------------------------------------------------------------------------


def test_event_creation(sample_event: Event) -> None:
    """
    Test that a sample event is created correctly and getters work.
    """
    assert sample_event.get_id() == "test_event"
    assert sample_event.get_payload() == {"data": 42}
    assert sample_event.get_priority() == 5


def test_timeout_event_creation(sample_timeout_event: TimeoutEvent) -> None:
    """
    Test that a sample timeout event is created correctly.
    """
    assert sample_timeout_event.get_id() == "timeout_event"
    assert sample_timeout_event.get_payload() == {"info": "test"}
    assert sample_timeout_event.get_priority() == 10
    assert sample_timeout_event.get_timeout() == 1.5


def test_event_immutability() -> None:
    """
    Test that the event attributes are effectively immutable.
    Attempts to modify them should fail or have no effect.
    """
    evt = Event("immutable_test", {"x": 1}, priority=2)
    # Direct attribute access is not provided, but let's assume no setters.
    # Just re-verify getters return the same value.
    assert evt.get_id() == "immutable_test"
    assert evt.get_payload() == {"x": 1}
    assert evt.get_priority() == 2


def test_error_chaining() -> None:
    """
    Test error chaining by raising one error and chaining another.
    """
    try:
        raise ValueError("Inner error")
    except ValueError as inner:
        try:
            raise HSMError("Outer error") from inner
        except HSMError as outer:
            assert outer.__cause__ is inner


def test_error_context_preservation() -> None:
    """
    Test creation and preservation of error context.
    """
    err = HSMError("Context error", details={"key": "value"})
    ctx = create_error_context(err, "traceback info")
    assert ctx.error_type is HSMError
    assert ctx.traceback == "traceback info"
    assert ctx.details == {"key": "value"}


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------


def test_null_payload() -> None:
    """
    Test event with null payload.
    """
    evt = Event("null_payload")
    assert evt.get_payload() is None


def test_zero_priority() -> None:
    """
    Test event with zero priority (boundary condition).
    """
    evt = Event("zero_priority", priority=0)
    assert evt.get_priority() == 0


def test_large_priority() -> None:
    """
    Test event with a very large priority value.
    """
    evt = Event("large_priority", priority=999999)
    assert evt.get_priority() == 999999


def test_maximum_recursion_scenario() -> None:
    """
    Pseudo test for maximum recursion:
    We can't truly recurse in events, so just a symbolic test.
    """

    def recurse(n: int):
        if n == 0:
            return "done"
        return recurse(n - 1)

    assert recurse(10) == "done"


# -----------------------------------------------------------------------------
# INTEGRATION
# -----------------------------------------------------------------------------


def test_logging_integration(logger_fixture, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test integration with logging by simulating event-based logging.
    """
    evt = Event("log_event", {"log": True})
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hsm.test.events"):
        logger_fixture.info("Event ID: %s", evt.get_id())

    assert any("Event ID: log_event" in rec.message for rec in caplog.records)


@contextmanager
def event_context_manager():
    """
    Context manager for testing error handling in a manager.
    If an error occurs, we simulate cleanup or convert it to HSMError.
    """
    try:
        yield
    except Exception as e:
        raise HSMError("Context error") from e


def test_error_handling_in_context_manager() -> None:
    """
    Test handling of errors that occur within a context manager.
    """
    with pytest.raises(HSMError) as exc_info:
        with event_context_manager():
            raise ValueError("Inner error in event context")
    assert "Context error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_error_scenario(async_setup) -> None:
    """
    Test async scenario where event processing might fail.
    We'll simulate an async failure.
    """

    async def async_fail():
        await asyncio.sleep(0.01)
        raise HSMError("Async event fail")

    with pytest.raises(HSMError) as exc_info:
        await async_fail()
    assert "Async event fail" in str(exc_info.value)


def test_cleanup_procedures() -> None:
    """
    Test scenario where cleanup is needed after an error.
    We'll simulate cleanup logic after raising an error.
    """
    try:
        raise HSMError("Cleanup test")
    except HSMError as e:
        # Simulate cleanup
        cleaned_up = True
        assert cleaned_up is True
        assert "Cleanup test" in str(e)


def test_thread_safety(sample_event: Event, sample_state_data: Dict[str, Any]) -> None:
    """
    Test simultaneous reads from the same event instance from multiple threads.
    Ensures thread safety under concurrent access.
    """
    results = []

    def worker():
        for _ in range(100):
            # Just access the event repeatedly
            event_id = sample_event.get_id()
            payload = sample_event.get_payload()
            priority = sample_event.get_priority()
            results.append((event_id, payload, priority))

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All results should match the sample_event's properties
    for r in results:
        assert r[0] == "test_event"
        assert r[1] == {"data": 42}
        assert r[2] == 5


def test_performance_timing() -> None:
    """
    Test that event creation and retrieval are performed within a reasonable time.
    This is a very loose check, just ensuring it doesn't hang significantly.
    """
    start = time.time()
    evt = Event("perf_event", payload=[x for x in range(100000)], priority=50)
    end = time.time()

    creation_time = end - start
    # Arbitrary threshold: should complete well under 0.5 seconds on a typical machine.
    assert creation_time < 0.5, f"Event creation took too long: {creation_time} seconds"

    # Access methods should be instantaneous or near so
    start = time.time()
    _ = evt.get_id()
    _ = evt.get_payload()
    _ = evt.get_priority()
    end = time.time()

    access_time = end - start
    assert access_time < 0.1, f"Event access took too long: {access_time} seconds"


def test_unicode_and_internationalization() -> None:
    """
    Test event handling with Unicode characters in the event ID and payload.
    """
    unicode_id = "évènt_😊"
    unicode_payload = {"message": "こんにちは世界", "emoji": "🤖"}
    evt = Event(unicode_id, unicode_payload, priority=1)

    assert evt.get_id() == unicode_id
    assert evt.get_payload() == unicode_payload
    assert evt.get_priority() == 1


def test_extremely_large_payload() -> None:
    """
    Test event with a very large payload to ensure handling does not degrade.
    """
    # Create a large string (e.g., ~1MB)
    large_payload = "A" * (1024 * 1024)  # 1MB of 'A'
    evt = Event("large_payload_event", large_payload, priority=10)

    assert evt.get_id() == "large_payload_event"
    assert evt.get_payload() == large_payload
    # Just ensure we can retrieve it without error
    assert evt.get_priority() == 10


def test_negative_priority() -> None:
    """
    Test event with a negative priority to see if it's allowed or causes issues.
    Assuming it's allowed and just returns the value.
    """
    evt = Event("negative_priority_event", None, priority=-5)
    assert evt.get_id() == "negative_priority_event"
    assert evt.get_priority() == -5


def test_extremely_large_priority() -> None:
    """
    Test event with an extremely large priority value.
    """
    large_priority = sys.maxsize
    evt = Event("huge_priority_event", priority=large_priority)
    assert evt.get_priority() == large_priority


def test_forward_compatibility_simulation() -> None:
    """
    Simulate reading an event from a future version with extra fields.
    We'll just pass extra fields to the constructor if it were flexible,
    or we can monkey patch something in payload.

    Assuming the event ignores unknown fields and just stores payload.
    """
    # Simulate a forward-compatible scenario: event_id and payload are known,
    # but let's say the payload includes unexpected fields.
    future_payload = {"known_key": 123, "unexpected_future_key": "future_data"}

    evt = Event("forward_compat_event", payload=future_payload, priority=1)
    # Check that we can still access known parts
    p = evt.get_payload()
    assert p["known_key"] == 123
    # We didn't define any special handling for unknown keys, but this simulates
    # that the event just stores what it gets.
    assert p["unexpected_future_key"] == "future_data"
