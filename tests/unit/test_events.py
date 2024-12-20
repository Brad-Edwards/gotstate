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
