"""Tests for gotstate.core.event module."""

import icontract
import pytest

from gotstate.core.event import (
    CallEvent,
    ChangeEvent,
    CompletionEvent,
    Event,
    EventKind,
    EventPriority,
    EventQueue,
    SignalEvent,
    TimeEvent,
)
from gotstate.exceptions import EventQueueFullError


class TestEvent:
    """Tests for the base Event class."""

    def test_create_event(self):
        e = Event("click", EventKind.SIGNAL)
        assert e.name == "click"
        assert e.kind == EventKind.SIGNAL
        assert e.priority == EventPriority.NORMAL
        assert isinstance(e.event_id, str)
        assert len(e.event_id) > 0
        assert e.timestamp > 0
        assert not e.is_consumed
        assert e.data == {}

    def test_create_event_with_data(self):
        e = Event("ev", EventKind.SIGNAL, data={"key": "value"})
        assert e.data == {"key": "value"}

    def test_event_data_immutability(self):
        original = {"key": "value"}
        e = Event("ev", EventKind.SIGNAL, data=original)
        original["key"] = "changed"
        assert e.data["key"] == "value"

    def test_event_name_contract(self):
        with pytest.raises(icontract.ViolationError):
            Event("", EventKind.SIGNAL)

    def test_event_kind_contract(self):
        with pytest.raises(icontract.ViolationError):
            Event("ev", "not_a_kind")  # type: ignore

    def test_event_priority_contract(self):
        with pytest.raises(icontract.ViolationError):
            Event("ev", EventKind.SIGNAL, "not_a_priority")  # type: ignore

    def test_consume(self):
        e = Event("ev", EventKind.SIGNAL)
        assert not e.is_consumed
        e.consume()
        assert e.is_consumed

    def test_event_ordering_by_priority(self):
        high = Event("h", EventKind.SIGNAL, EventPriority.HIGH)
        normal = Event("n", EventKind.SIGNAL, EventPriority.NORMAL)
        low = Event("l", EventKind.SIGNAL, EventPriority.LOW)
        events = [low, normal, high]
        events.sort()
        assert events[0].priority == EventPriority.HIGH
        assert events[1].priority == EventPriority.NORMAL
        assert events[2].priority == EventPriority.LOW

    def test_event_equality_by_id(self):
        e1 = Event("ev", EventKind.SIGNAL)
        e2 = Event("ev", EventKind.SIGNAL)
        assert e1 != e2  # Different event IDs
        assert e1 == e1

    def test_event_hash(self):
        e = Event("ev", EventKind.SIGNAL)
        assert hash(e) == hash(e.event_id)

    def test_repr(self):
        e = Event("click", EventKind.SIGNAL, EventPriority.HIGH)
        assert "click" in repr(e)
        assert "SIGNAL" in repr(e)
        assert "HIGH" in repr(e)


class TestSignalEvent:
    def test_create(self):
        e = SignalEvent("sig")
        assert e.kind == EventKind.SIGNAL
        assert e.priority == EventPriority.NORMAL

    def test_with_priority(self):
        e = SignalEvent("sig", EventPriority.HIGH)
        assert e.priority == EventPriority.HIGH


class TestCallEvent:
    def test_create(self):
        e = CallEvent("call")
        assert e.kind == EventKind.CALL
        assert e.operation is None

    def test_with_operation(self):
        op = lambda: 42
        e = CallEvent("call", operation=op)
        assert e.operation is op

    def test_result(self):
        e = CallEvent("call")
        assert e.result is None
        e.result = 42
        assert e.result == 42


class TestTimeEvent:
    def test_create_with_duration(self):
        e = TimeEvent("timeout", duration=5.0)
        assert e.kind == EventKind.TIME
        assert e.duration == 5.0
        assert not e.is_cancelled

    def test_negative_duration_contract(self):
        with pytest.raises(icontract.ViolationError):
            TimeEvent("timeout", duration=-1.0)

    def test_cancel(self):
        e = TimeEvent("timeout", duration=5.0)
        e.cancel()
        assert e.is_cancelled


class TestChangeEvent:
    def test_create(self):
        condition = lambda: True
        e = ChangeEvent("change", condition)
        assert e.kind == EventKind.CHANGE
        assert e.condition is condition

    def test_evaluate_true(self):
        e = ChangeEvent("change", lambda: True)
        assert e.evaluate() is True

    def test_evaluate_false(self):
        e = ChangeEvent("change", lambda: False)
        assert e.evaluate() is False

    def test_evaluate_detects_change(self):
        values = [False, True]
        idx = [0]

        def condition():
            val = values[idx[0]]
            idx[0] = min(idx[0] + 1, len(values) - 1)
            return val

        e = ChangeEvent("change", condition)
        assert e.evaluate() is False  # False, no change to True
        assert e.evaluate() is True   # Changed to True

    def test_non_callable_condition_contract(self):
        with pytest.raises(icontract.ViolationError):
            ChangeEvent("change", "not_callable")  # type: ignore


class TestCompletionEvent:
    def test_create(self):
        e = CompletionEvent("done", "state_a")
        assert e.kind == EventKind.COMPLETION
        assert e.priority == EventPriority.HIGH
        assert e.source_state_name == "state_a"


class TestEventQueue:
    """Tests for the EventQueue class."""

    def test_create_queue(self):
        q = EventQueue()
        assert q.is_empty
        assert q.size == 0
        assert q.max_size == 1000

    def test_custom_max_size(self):
        q = EventQueue(max_size=10)
        assert q.max_size == 10

    def test_invalid_max_size_contract(self):
        with pytest.raises(icontract.ViolationError):
            EventQueue(max_size=0)
        with pytest.raises(icontract.ViolationError):
            EventQueue(max_size=-1)

    def test_enqueue_dequeue(self):
        q = EventQueue()
        e = Event("ev", EventKind.SIGNAL)
        q.enqueue(e)
        assert q.size == 1
        assert not q.is_empty
        result = q.dequeue()
        assert result is e
        assert q.is_empty

    def test_priority_ordering(self):
        q = EventQueue()
        low = Event("low", EventKind.SIGNAL, EventPriority.LOW)
        high = Event("high", EventKind.SIGNAL, EventPriority.HIGH)
        normal = Event("normal", EventKind.SIGNAL, EventPriority.NORMAL)

        q.enqueue(low)
        q.enqueue(high)
        q.enqueue(normal)

        assert q.dequeue().priority == EventPriority.HIGH
        assert q.dequeue().priority == EventPriority.NORMAL
        assert q.dequeue().priority == EventPriority.LOW

    def test_dequeue_empty_returns_none(self):
        q = EventQueue()
        assert q.dequeue() is None

    def test_peek(self):
        q = EventQueue()
        e = Event("ev", EventKind.SIGNAL)
        q.enqueue(e)
        assert q.peek() is e
        assert q.size == 1  # Not removed

    def test_peek_empty_returns_none(self):
        q = EventQueue()
        assert q.peek() is None

    def test_clear(self):
        q = EventQueue()
        for i in range(5):
            q.enqueue(Event(f"ev{i}", EventKind.SIGNAL))
        q.clear()
        assert q.is_empty

    def test_cancel(self):
        q = EventQueue()
        e1 = Event("e1", EventKind.SIGNAL)
        e2 = Event("e2", EventKind.SIGNAL)
        q.enqueue(e1)
        q.enqueue(e2)
        assert q.cancel(e1.event_id)
        assert q.size == 1
        assert q.dequeue() is e2

    def test_cancel_nonexistent(self):
        q = EventQueue()
        assert not q.cancel("nonexistent")

    def test_deferred_events(self):
        q = EventQueue()
        deferred = Event("deferred", EventKind.SIGNAL, EventPriority.DEFER)
        q.enqueue(deferred)
        assert q.size == 0  # Deferred events are not in main queue

        flushed = q.flush_deferred()
        assert len(flushed) == 1
        assert q.size == 1  # Now in main queue

    def test_queue_full_raises(self):
        q = EventQueue(max_size=2)
        q.enqueue(Event("e1", EventKind.SIGNAL))
        q.enqueue(Event("e2", EventKind.SIGNAL))
        with pytest.raises(EventQueueFullError):
            q.enqueue(Event("e3", EventKind.SIGNAL))

    def test_len(self):
        q = EventQueue()
        assert len(q) == 0
        q.enqueue(Event("ev", EventKind.SIGNAL))
        assert len(q) == 1

    def test_enqueue_non_event_contract(self):
        q = EventQueue()
        with pytest.raises(icontract.ViolationError):
            q.enqueue("not_an_event")  # type: ignore

    def test_repr(self):
        q = EventQueue(max_size=50)
        assert "50" in repr(q)
