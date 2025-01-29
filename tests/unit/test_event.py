"""Unit tests for the Event class and its subclasses.

Tests the event processing and queue management functionality.
"""

import unittest
from unittest.mock import Mock, patch

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


class TestEvent(unittest.TestCase):
    """Test cases for the base Event class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_id = "test_event"
        self.event_kind = EventKind.SIGNAL
        self.event_priority = EventPriority.NORMAL
        self.event_data = {"key": "value"}
        self.event_timeout = 1000  # milliseconds

    def test_event_creation(self):
        """Test that an Event can be created with valid parameters."""
        event = Event(
            event_id=self.event_id,
            kind=self.event_kind,
            priority=self.event_priority,
            data=self.event_data,
            timeout=self.event_timeout,
        )

        self.assertEqual(event.id, self.event_id)
        self.assertEqual(event.kind, self.event_kind)
        self.assertEqual(event.priority, self.event_priority)
        self.assertEqual(event.data, self.event_data)
        self.assertEqual(event.timeout, self.event_timeout)
        self.assertFalse(event.consumed)
        self.assertFalse(event.cancelled)

    def test_event_invalid_id(self):
        """Test that Event creation fails with invalid ID."""
        with self.assertRaises(ValueError):
            Event(event_id=None, kind=self.event_kind, priority=self.event_priority)

        with self.assertRaises(ValueError):
            Event(event_id="", kind=self.event_kind, priority=self.event_priority)

    def test_event_invalid_kind(self):
        """Test that Event creation fails with invalid kind."""
        with self.assertRaises(ValueError):
            Event(event_id=self.event_id, kind=None, priority=self.event_priority)

    def test_event_invalid_priority(self):
        """Test that Event creation fails with invalid priority."""
        with self.assertRaises(ValueError):
            Event(event_id=self.event_id, kind=self.event_kind, priority=None)

    def test_event_invalid_timeout(self):
        """Test that Event creation fails with negative timeout."""
        with self.assertRaises(ValueError):
            Event(event_id=self.event_id, kind=self.event_kind, priority=self.event_priority, timeout=-1)

    def test_event_data_immutability(self):
        """Test that event data is immutable."""
        original_data = {"key": "value"}
        event = Event(event_id=self.event_id, kind=self.event_kind, priority=self.event_priority, data=original_data)

        # Verify that modifying the original data doesn't affect the event
        original_data["key"] = "modified"
        self.assertEqual(event.data["key"], "value")

        # Verify that modifying the returned data doesn't affect the event
        event_data = event.data
        event_data["key"] = "modified"
        self.assertEqual(event.data["key"], "value")

    def test_event_consumption(self):
        """Test event consumption functionality."""
        event = Event(event_id=self.event_id, kind=self.event_kind, priority=self.event_priority)

        self.assertFalse(event.consumed)
        event.consume()
        self.assertTrue(event.consumed)

    def test_event_cancellation(self):
        """Test event cancellation functionality."""
        event = Event(event_id=self.event_id, kind=self.event_kind, priority=self.event_priority)

        self.assertFalse(event.cancelled)
        event.cancel()
        self.assertTrue(event.cancelled)


class TestSignalEvent(unittest.TestCase):
    """Test cases for the SignalEvent class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_id = "test_signal"
        self.event_priority = EventPriority.NORMAL
        self.event_data = {"payload": "test_data"}
        self.event_timeout = 1000

    def test_signal_event_creation(self):
        """Test that a SignalEvent can be created with valid parameters."""
        event = SignalEvent(
            event_id=self.event_id, priority=self.event_priority, data=self.event_data, timeout=self.event_timeout
        )

        self.assertEqual(event.id, self.event_id)
        self.assertEqual(event.kind, EventKind.SIGNAL)
        self.assertEqual(event.priority, self.event_priority)
        self.assertEqual(event.data, self.event_data)
        self.assertEqual(event.timeout, self.event_timeout)
        self.assertFalse(event.consumed)
        self.assertFalse(event.cancelled)

    def test_signal_event_broadcast(self):
        """Test signal event broadcast functionality."""
        event = SignalEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        # Verify that consuming a broadcast signal doesn't prevent further consumption
        event.consume()
        self.assertTrue(event.consumed)
        self.assertTrue(event.can_consume())  # SignalEvents can be consumed multiple times

    def test_signal_event_ordering(self):
        """Test that signal events maintain proper ordering."""
        high_priority = SignalEvent(
            event_id="high_priority",
            priority=EventPriority.HIGH,
        )
        normal_priority = SignalEvent(
            event_id="normal_priority",
            priority=EventPriority.NORMAL,
        )
        low_priority = SignalEvent(
            event_id="low_priority",
            priority=EventPriority.LOW,
        )

        # Verify priority ordering
        self.assertTrue(high_priority.priority.value < normal_priority.priority.value)
        self.assertTrue(normal_priority.priority.value < low_priority.priority.value)

    def test_signal_event_listener_management(self):
        """Test signal event listener management functionality."""
        event = SignalEvent(event_id=self.event_id, priority=self.event_priority)

        # Initially no listeners
        self.assertEqual(len(event.listeners), 0)

        # Add listeners
        event.add_listener("listener1")
        event.add_listener("listener2")
        self.assertEqual(len(event.listeners), 2)
        self.assertIn("listener1", event.listeners)
        self.assertIn("listener2", event.listeners)

        # Add duplicate listener
        event.add_listener("listener1")
        self.assertEqual(len(event.listeners), 2)  # Set semantics prevent duplicates

        # Remove listener
        event.remove_listener("listener1")
        self.assertEqual(len(event.listeners), 1)
        self.assertNotIn("listener1", event.listeners)
        self.assertIn("listener2", event.listeners)

        # Remove non-existent listener
        event.remove_listener("listener3")  # Should not raise error
        self.assertEqual(len(event.listeners), 1)

        # Verify listeners set is copied
        listeners = event.listeners
        listeners.add("listener3")
        self.assertEqual(len(event.listeners), 1)  # Original set unchanged


class TestCallEvent(unittest.TestCase):
    """Test cases for the CallEvent class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_id = "test_call"
        self.event_priority = EventPriority.NORMAL
        self.event_data = {"method": "test_method", "args": ["arg1", "arg2"], "kwargs": {"key": "value"}}
        self.event_timeout = 1000

    def test_call_event_creation(self):
        """Test that a CallEvent can be created with valid parameters."""
        event = CallEvent(
            event_id=self.event_id, priority=self.event_priority, data=self.event_data, timeout=self.event_timeout
        )

        self.assertEqual(event.id, self.event_id)
        self.assertEqual(event.kind, EventKind.CALL)
        self.assertEqual(event.priority, self.event_priority)
        self.assertEqual(event.data, self.event_data)
        self.assertEqual(event.timeout, self.event_timeout)
        self.assertFalse(event.consumed)
        self.assertFalse(event.cancelled)
        self.assertIsNone(event.return_value)

    def test_call_event_single_consumption(self):
        """Test that call events can only be consumed once."""
        event = CallEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        self.assertTrue(event.can_consume())  # Initial state
        event.consume()
        self.assertFalse(event.can_consume())  # Cannot consume again

    def test_call_event_return_value(self):
        """Test call event return value handling."""
        event = CallEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        # Initially no return value
        self.assertIsNone(event.return_value)

        # Set return value
        test_return = {"result": "success"}
        event.set_return_value(test_return)
        self.assertEqual(event.return_value, test_return)

        # Verify return value is copied
        test_return["result"] = "modified"
        self.assertEqual(event.return_value["result"], "success")

    def test_call_event_parameter_validation(self):
        """Test call event parameter validation."""
        # Missing required method parameter
        with self.assertRaises(ValueError):
            CallEvent(event_id=self.event_id, priority=self.event_priority, data={"args": [], "kwargs": {}})

        # Invalid args type
        with self.assertRaises(ValueError):
            CallEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"method": "test", "args": "not_a_list", "kwargs": {}},
            )

        # Invalid kwargs type
        with self.assertRaises(ValueError):
            CallEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"method": "test", "args": [], "kwargs": "not_a_dict"},
            )


class TestTimeEvent(unittest.TestCase):
    """Test cases for the TimeEvent class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_id = "test_time"
        self.event_priority = EventPriority.NORMAL
        self.event_data = {
            "type": "after",  # or "at"
            "time": 1000,  # milliseconds for "after", timestamp for "at"
            "repeat": False,
        }
        self.event_timeout = None  # TimeEvents manage their own timing

    def test_time_event_creation(self):
        """Test that a TimeEvent can be created with valid parameters."""
        event = TimeEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        self.assertEqual(event.id, self.event_id)
        self.assertEqual(event.kind, EventKind.TIME)
        self.assertEqual(event.priority, self.event_priority)
        self.assertEqual(event.data, self.event_data)
        self.assertIsNone(event.timeout)  # TimeEvents manage their own timing
        self.assertFalse(event.consumed)
        self.assertFalse(event.cancelled)
        self.assertFalse(event.is_expired)

    def test_time_event_validation(self):
        """Test time event parameter validation."""
        # Missing type parameter
        with self.assertRaises(ValueError):
            TimeEvent(event_id=self.event_id, priority=self.event_priority, data={"time": 1000})

        # Invalid type value
        with self.assertRaises(ValueError):
            TimeEvent(event_id=self.event_id, priority=self.event_priority, data={"type": "invalid", "time": 1000})

        # Missing time parameter
        with self.assertRaises(ValueError):
            TimeEvent(event_id=self.event_id, priority=self.event_priority, data={"type": "after"})

        # Invalid time value (negative)
        with self.assertRaises(ValueError):
            TimeEvent(event_id=self.event_id, priority=self.event_priority, data={"type": "after", "time": -1000})

    def test_time_event_expiration(self):
        """Test time event expiration functionality."""
        event = TimeEvent(event_id=self.event_id, priority=self.event_priority, data={"type": "after", "time": 1000})

        self.assertFalse(event.is_expired)
        event.expire()
        self.assertTrue(event.is_expired)
        self.assertTrue(event.consumed)  # Expiration implies consumption

    def test_time_event_repeat(self):
        """Test repeating time event functionality."""
        event = TimeEvent(
            event_id=self.event_id, priority=self.event_priority, data={"type": "after", "time": 1000, "repeat": True}
        )

        self.assertTrue(event.can_consume())
        event.consume()
        self.assertTrue(event.can_consume())  # Repeating events can be consumed multiple times

        # Non-repeating event
        event = TimeEvent(
            event_id=self.event_id, priority=self.event_priority, data={"type": "after", "time": 1000, "repeat": False}
        )

        self.assertTrue(event.can_consume())
        event.consume()
        self.assertFalse(event.can_consume())  # Non-repeating events can only be consumed once


class TestChangeEvent(unittest.TestCase):
    """Test cases for the ChangeEvent class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_id = "test_change"
        self.event_priority = EventPriority.NORMAL
        self.event_data = {
            "condition": "value_changed",
            "target": "test_variable",
            "old_value": 1,
            "new_value": 2,
            "track_history": True,
        }
        self.event_timeout = None

    def test_change_event_creation(self):
        """Test that a ChangeEvent can be created with valid parameters."""
        event = ChangeEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        self.assertEqual(event.id, self.event_id)
        self.assertEqual(event.kind, EventKind.CHANGE)
        self.assertEqual(event.priority, self.event_priority)
        self.assertEqual(event.data, self.event_data)
        self.assertFalse(event.consumed)
        self.assertFalse(event.cancelled)
        self.assertEqual(len(event.history), 1)  # Initial change is recorded

    def test_change_event_validation(self):
        """Test change event parameter validation."""
        # Missing condition parameter
        with self.assertRaises(ValueError):
            ChangeEvent(event_id=self.event_id, priority=self.event_priority, data={"target": "test_variable"})

        # Missing target parameter
        with self.assertRaises(ValueError):
            ChangeEvent(event_id=self.event_id, priority=self.event_priority, data={"condition": "value_changed"})

        # Invalid condition value
        with self.assertRaises(ValueError):
            ChangeEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"condition": "invalid_condition", "target": "test_variable"},
            )

    def test_change_event_history(self):
        """Test change event history tracking."""
        event = ChangeEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        # Initial change is recorded
        self.assertEqual(len(event.history), 1)
        initial_change = event.history[0]
        self.assertEqual(initial_change["old_value"], 1)
        self.assertEqual(initial_change["new_value"], 2)

        # Record another change
        event.record_change(2, 3)
        self.assertEqual(len(event.history), 2)
        latest_change = event.history[-1]
        self.assertEqual(latest_change["old_value"], 2)
        self.assertEqual(latest_change["new_value"], 3)

        # Test history disabled
        event = ChangeEvent(
            event_id=self.event_id, priority=self.event_priority, data={**self.event_data, "track_history": False}
        )

        self.assertEqual(len(event.history), 0)  # No history when tracking disabled
        event.record_change(1, 2)
        self.assertEqual(len(event.history), 0)  # Still no history

    def test_change_event_consumption(self):
        """Test change event consumption behavior."""
        event = ChangeEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        self.assertTrue(event.can_consume())
        event.consume()
        self.assertFalse(event.can_consume())  # Change events can only be consumed once


class TestCompletionEvent(unittest.TestCase):
    """Test cases for the CompletionEvent class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_id = "test_completion"
        self.event_priority = EventPriority.NORMAL
        self.event_data = {
            "state_id": "test_state",
            "region_id": "test_region",
            "completion_type": "do_activity",  # or "final"
            "result": "success",
        }
        self.event_timeout = None

    def test_completion_event_creation(self):
        """Test that a CompletionEvent can be created with valid parameters."""
        event = CompletionEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        self.assertEqual(event.id, self.event_id)
        self.assertEqual(event.kind, EventKind.COMPLETION)
        self.assertEqual(event.priority, self.event_priority)
        self.assertEqual(event.data, self.event_data)
        self.assertFalse(event.consumed)
        self.assertFalse(event.cancelled)

    def test_completion_event_validation(self):
        """Test completion event parameter validation."""
        # Missing state_id parameter
        with self.assertRaises(ValueError):
            CompletionEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"region_id": "test_region", "completion_type": "do_activity"},
            )

        # Missing region_id parameter
        with self.assertRaises(ValueError):
            CompletionEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"state_id": "test_state", "completion_type": "do_activity"},
            )

        # Missing completion_type parameter
        with self.assertRaises(ValueError):
            CompletionEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"state_id": "test_state", "region_id": "test_region"},
            )

        # Invalid completion_type value
        with self.assertRaises(ValueError):
            CompletionEvent(
                event_id=self.event_id,
                priority=self.event_priority,
                data={"state_id": "test_state", "region_id": "test_region", "completion_type": "invalid"},
            )

    def test_completion_event_consumption(self):
        """Test completion event consumption behavior."""
        event = CompletionEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)

        self.assertTrue(event.can_consume())
        event.consume()
        self.assertFalse(event.can_consume())  # Completion events can only be consumed once

    def test_completion_event_result_handling(self):
        """Test completion event result handling."""
        # Test with result
        event = CompletionEvent(event_id=self.event_id, priority=self.event_priority, data=self.event_data)
        self.assertEqual(event.result, "success")

        # Test without result
        event = CompletionEvent(
            event_id=self.event_id,
            priority=self.event_priority,
            data={"state_id": "test_state", "region_id": "test_region", "completion_type": "do_activity"},
        )
        self.assertIsNone(event.result)


class TestEventQueue(unittest.TestCase):
    """Test cases for the EventQueue class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.queue = EventQueue()
        self.event_data = {"test": "data"}

    def test_event_queue_creation(self):
        """Test that an EventQueue can be created."""
        self.assertEqual(len(self.queue), 0)
        self.assertFalse(self.queue.is_processing)

    def test_event_enqueue_dequeue(self):
        """Test basic event enqueue and dequeue operations."""
        # Create events with different priorities
        high_priority = Event(event_id="high", kind=EventKind.SIGNAL, priority=EventPriority.HIGH, data=self.event_data)
        normal_priority = Event(
            event_id="normal", kind=EventKind.SIGNAL, priority=EventPriority.NORMAL, data=self.event_data
        )
        low_priority = Event(event_id="low", kind=EventKind.SIGNAL, priority=EventPriority.LOW, data=self.event_data)

        # Enqueue events in random order
        self.queue.enqueue(normal_priority)
        self.queue.enqueue(low_priority)
        self.queue.enqueue(high_priority)

        self.assertEqual(len(self.queue), 3)

        # Verify dequeue order respects priority
        self.assertEqual(self.queue.dequeue(), high_priority)
        self.assertEqual(self.queue.dequeue(), normal_priority)
        self.assertEqual(self.queue.dequeue(), low_priority)

        self.assertEqual(len(self.queue), 0)

    def test_event_queue_cancellation(self):
        """Test event cancellation in queue."""
        event = Event(event_id="test", kind=EventKind.SIGNAL, priority=EventPriority.NORMAL, data=self.event_data)

        self.queue.enqueue(event)
        self.assertEqual(len(self.queue), 1)

        event.cancel()
        self.assertEqual(len(self.queue), 0)  # Cancelled events are removed

    def test_event_queue_timeout(self):
        """Test event timeout handling."""
        event = Event(
            event_id="test",
            kind=EventKind.SIGNAL,
            priority=EventPriority.NORMAL,
            data=self.event_data,
            timeout=0,  # Immediate timeout
        )

        self.queue.enqueue(event)
        self.assertEqual(len(self.queue), 0)  # Timed out events are not queued

    def test_event_queue_processing(self):
        """Test event processing state."""
        event = Event(event_id="test", kind=EventKind.SIGNAL, priority=EventPriority.NORMAL, data=self.event_data)

        self.queue.enqueue(event)
        self.assertFalse(self.queue.is_processing)

        self.queue.start_processing()
        self.assertTrue(self.queue.is_processing)

        self.queue.stop_processing()
        self.assertFalse(self.queue.is_processing)

    def test_event_queue_filtering(self):
        """Test event filtering functionality."""
        signal_event = Event(
            event_id="signal", kind=EventKind.SIGNAL, priority=EventPriority.NORMAL, data=self.event_data
        )
        call_event = Event(event_id="call", kind=EventKind.CALL, priority=EventPriority.NORMAL, data=self.event_data)

        self.queue.enqueue(signal_event)
        self.queue.enqueue(call_event)

        # Filter by event kind
        signal_events = self.queue.filter(lambda e: e.kind == EventKind.SIGNAL)
        self.assertEqual(len(signal_events), 1)
        self.assertEqual(signal_events[0], signal_event)

        call_events = self.queue.filter(lambda e: e.kind == EventKind.CALL)
        self.assertEqual(len(call_events), 1)
        self.assertEqual(call_events[0], call_event)


if __name__ == "__main__":
    unittest.main()
