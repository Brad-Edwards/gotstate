import time
import unittest
from threading import Thread
from typing import Any, Dict

from gotstate.core.machine.machine_monitor import MachineMonitor


class TestMachineMonitor(unittest.TestCase):
    """Test cases for MachineMonitor class.

    Tests verify:
    1. Metrics management
    2. Event history tracking
    3. Event querying and filtering
    4. Thread safety
    5. Time-based operations
    6. Binary search functionality
    """

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = MachineMonitor()

    def test_initial_state(self):
        """Test initial state of monitor.

        Verifies:
        1. Empty metrics
        2. Empty history
        3. Zero event count
        4. Locks initialized
        """
        self.assertEqual(len(self.monitor.metrics), 0)
        self.assertEqual(len(self.monitor.history), 0)
        self.assertEqual(self.monitor.event_count, 0)
        self.assertIsNotNone(self.monitor._metrics_lock)
        self.assertIsNotNone(self.monitor._history_lock)

    def test_update_metric(self):
        """Test metric updates.

        Verifies:
        1. New metric creation
        2. Existing metric update
        3. Multiple updates
        """
        # Test new metric
        self.monitor.update_metric("test_metric", 5)
        self.assertEqual(self.monitor.get_metric("test_metric"), 5)

        # Test update existing
        self.monitor.update_metric("test_metric", 3)
        self.assertEqual(self.monitor.get_metric("test_metric"), 8)

        # Test multiple metrics
        self.monitor.update_metric("another_metric", 10)
        self.assertEqual(self.monitor.get_metric("another_metric"), 10)
        self.assertEqual(len(self.monitor.metrics), 2)

    def test_get_metric_nonexistent(self):
        """Test getting non-existent metric.

        Verifies proper error handling for missing metrics.
        """
        with self.assertRaises(KeyError):
            self.monitor.get_metric("nonexistent")

    def test_track_event(self):
        """Test event tracking.

        Verifies:
        1. Event added to history
        2. Event count updated
        3. Event data copied
        4. Indices updated
        """
        event = {"type": "test_event", "timestamp": time.time(), "data": "test_data"}

        self.monitor.track_event(event)

        self.assertEqual(self.monitor.event_count, 1)
        self.assertEqual(len(self.monitor.history), 1)
        self.assertEqual(self.monitor.history[0]["type"], "test_event")

        # Verify event data is copied
        event["data"] = "modified"
        self.assertEqual(self.monitor.history[0]["data"], "test_data")

    def test_query_events_by_type(self):
        """Test event querying by type.

        Verifies:
        1. Type filtering works
        2. Multiple events handled
        3. No matches handled
        """
        # Add test events
        events = [
            {"type": "type1", "data": "data1", "timestamp": time.time()},
            {"type": "type2", "data": "data2", "timestamp": time.time()},
            {"type": "type1", "data": "data3", "timestamp": time.time()},
        ]
        for event in events:
            self.monitor.track_event(event)

        # Query by type
        type1_events = self.monitor.query_events(event_type="type1")
        self.assertEqual(len(type1_events), 2)
        self.assertTrue(all(e["type"] == "type1" for e in type1_events))

        # Query non-existent type
        empty_events = self.monitor.query_events(event_type="nonexistent")
        self.assertEqual(len(empty_events), 0)

    def test_query_events_by_time(self):
        """Test event querying by time.

        Verifies:
        1. Time filtering works
        2. Events properly ordered
        3. Edge cases handled
        """
        # Add events with different timestamps
        now = time.time()
        events = [
            {"type": "test", "timestamp": now - 2, "data": "old"},
            {"type": "test", "timestamp": now - 1, "data": "middle"},
            {"type": "test", "timestamp": now, "data": "new"},
        ]
        for event in events:
            self.monitor.track_event(event)

        # Query with start time
        filtered_events = self.monitor.query_events(start_time=now - 1)
        self.assertEqual(len(filtered_events), 2)
        self.assertEqual(filtered_events[0]["data"], "middle")
        self.assertEqual(filtered_events[1]["data"], "new")

    def test_query_events_combined_filters(self):
        """Test event querying with combined filters.

        Verifies:
        1. Type and time filters combined
        2. Order preserved
        3. No matches handled
        """
        now = time.time()
        events = [
            {"type": "type1", "timestamp": now - 2, "data": "old1"},
            {"type": "type2", "timestamp": now - 1, "data": "old2"},
            {"type": "type1", "timestamp": now, "data": "new1"},
        ]
        for event in events:
            self.monitor.track_event(event)

        # Query with both filters
        filtered_events = self.monitor.query_events(event_type="type1", start_time=now - 1)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["data"], "new1")

    def test_metrics_thread_safety(self):
        """Test thread safety of metrics operations.

        Verifies:
        1. Concurrent updates handled
        2. No data races
        3. Final value correct
        """

        def update_metric():
            for _ in range(100):
                self.monitor.update_metric("counter", 1)

        threads = [Thread(target=update_metric) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(self.monitor.get_metric("counter"), 1000)

    def test_history_thread_safety(self):
        """Test thread safety of history operations.

        Verifies:
        1. Concurrent event tracking
        2. Event count accuracy
        3. No data corruption
        """

        def track_events():
            for i in range(100):
                event = {"type": "test", "timestamp": time.time(), "data": f"event_{i}"}
                self.monitor.track_event(event)

        threads = [Thread(target=track_events) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(self.monitor.event_count, 1000)
        self.assertEqual(len(self.monitor.history), 1000)

    def test_binary_search_time(self):
        """Test binary search for time index.

        Verifies:
        1. Exact matches found
        2. Nearest greater found
        3. Edge cases handled
        """
        now = time.time()
        events = [
            {"type": "test", "timestamp": now - 2},
            {"type": "test", "timestamp": now - 1},
            {"type": "test", "timestamp": now},
        ]
        for event in events:
            self.monitor.track_event(event)

        # Test exact match
        index = self.monitor._binary_search_time(now - 1)
        self.assertEqual(self.monitor._time_index[index][0], now - 1)

        # Test between values
        index = self.monitor._binary_search_time(now - 1.5)
        self.assertEqual(self.monitor._time_index[index][0], now - 1)

        # Test beyond range
        index = self.monitor._binary_search_time(now + 1)
        self.assertEqual(index, len(self.monitor._time_index))

    def test_clear_history(self):
        """Test history clearing.

        Verifies:
        1. Full clear works
        2. Time-based clear works
        3. Indices updated
        """
        # Add test events
        now = time.time()
        events = [
            {"type": "test", "timestamp": now - 2},
            {"type": "test", "timestamp": now - 1},
            {"type": "test", "timestamp": now},
        ]
        for event in events:
            self.monitor.track_event(event)

        # Test partial clear
        self.monitor.clear_history(before_time=now - 1)
        self.assertEqual(len(self.monitor.history), 2)
        self.assertEqual(self.monitor.event_count, 2)

        # Test full clear
        self.monitor.clear_history()
        self.assertEqual(len(self.monitor.history), 0)
        self.assertEqual(self.monitor.event_count, 0)
        self.assertEqual(len(self.monitor._event_index), 0)
        self.assertEqual(len(self.monitor._time_index), 0)

    def test_get_metrics_snapshot(self):
        """Test metrics snapshot.

        Verifies:
        1. All metrics included
        2. Values correct
        3. Snapshot isolated
        """
        # Add test metrics
        self.monitor.update_metric("metric1", 5)
        self.monitor.update_metric("metric2", 10)

        # Get snapshot
        snapshot = self.monitor.get_metrics_snapshot()
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot["metric1"], 5)
        self.assertEqual(snapshot["metric2"], 10)

        # Verify snapshot isolation
        snapshot["metric1"] = 100
        self.assertEqual(self.monitor.get_metric("metric1"), 5)


if __name__ == "__main__":
    unittest.main()
