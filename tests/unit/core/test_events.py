# tests/unit/test_events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


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
