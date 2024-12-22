# tests/unit/test_events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest


def test_event_init():
    from hsm.core.events import Event

    e = Event("MyEvent")
    assert e.name == "MyEvent"
    assert isinstance(e.metadata, dict)


def test_timeout_event_init():
    from hsm.core.events import TimeoutEvent

    e = TimeoutEvent("Timeout", 123.456)
    assert e.name == "Timeout"
    assert e.deadline == 123.456


def test_event_metadata():
    from hsm.core.events import Event

    e = Event("TestEvent")
    e.metadata["key"] = "value"
    assert e.metadata["key"] == "value"
    assert len(e.metadata) == 1


def test_event_metadata_isolation():
    from hsm.core.events import Event

    e1 = Event("Event1")
    e2 = Event("Event2")

    e1.metadata["key"] = "value1"
    e2.metadata["key"] = "value2"

    assert e1.metadata["key"] == "value1"
    assert e2.metadata["key"] == "value2"


def test_timeout_event_inheritance():
    from hsm.core.events import Event, TimeoutEvent

    e = TimeoutEvent("Timeout", 10.0)
    assert isinstance(e, Event)
    assert isinstance(e, TimeoutEvent)


def test_timeout_event_metadata():
    from hsm.core.events import TimeoutEvent

    e = TimeoutEvent("Timeout", 10.0)
    e.metadata["key"] = "value"
    assert e.metadata["key"] == "value"
    assert e.deadline == 10.0


def test_event_empty_metadata():
    from hsm.core.events import Event

    e = Event("TestEvent")
    assert isinstance(e.metadata, dict)
    assert len(e.metadata) == 0


def test_event_priority():
    from hsm.core.events import Event

    e = Event("TestEvent", priority=5)
    assert e.priority == 5


def test_event_comparison_by_priority():
    from hsm.core.events import Event

    high_priority = Event("High", priority=2)
    low_priority = Event("Low", priority=1)

    # Higher priority should come first
    assert high_priority < low_priority
    assert not low_priority < high_priority


def test_event_comparison_by_timestamp():
    import time

    from hsm.core.events import Event

    e1 = Event("First")
    time.sleep(0.001)  # Ensure different timestamps
    e2 = Event("Second")

    # Same priority (0), earlier timestamp should come first
    assert e1 < e2
    assert not e2 < e1


def test_event_equality():
    from hsm.core.events import Event

    e1 = Event("Test", priority=1)
    e2 = Event("Test", priority=1)
    e3 = Event("Test", priority=2)

    assert e1 != e2  # Different timestamps
    assert e1 != e3  # Different priorities
    assert e1 == e1  # Same object


def test_event_comparison_with_non_event():
    from hsm.core.events import Event

    e = Event("Test")
    # Less than should raise TypeError
    with pytest.raises(TypeError):
        _ = e < "not_an_event"

    # Equality should return False for non-Event types
    assert not (e == "not_an_event")
    assert e != "not_an_event"
