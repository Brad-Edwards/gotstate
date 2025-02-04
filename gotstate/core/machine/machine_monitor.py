import threading
from typing import Any, Dict, List, Optional


class MachineMonitor:
    """Monitors state machine execution.

    MachineMonitor implements introspection capabilities and
    collects runtime metrics.

    Class Invariants:
    1. Must track metrics
    2. Must handle events
    3. Must maintain history
    4. Must support queries

    Design Patterns:
    - Observer: Monitors changes
    - Strategy: Implements policies
    - Command: Encapsulates queries

    Threading/Concurrency Guarantees:
    1. Thread-safe monitoring
    2. Atomic updates
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) metric updates
    2. O(q) query execution where q is query complexity
    3. O(h) history tracking where h is history size
    """

    def __init__(self):
        """Initialize the machine monitor."""
        self._metrics = {}
        self._history = []
        self._event_count = 0
        self._metrics_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._event_index = {}  # Type -> List[event indices]
        self._time_index = []  # List of (timestamp, index) tuples

    @property
    def metrics(self) -> Dict[str, int]:
        """Get the current metrics.

        Returns:
            A copy of the metrics dictionary
        """
        with self._metrics_lock:
            return self._metrics.copy()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get the event history.

        Returns:
            A copy of the event history list
        """
        with self._history_lock:
            return self._history.copy()

    @property
    def event_count(self) -> int:
        """Get the total number of events tracked.

        Returns:
            The event count
        """
        with self._history_lock:
            return self._event_count

    def track_event(self, event: Dict[str, Any]) -> None:
        """Track a machine event.

        Args:
            event: Event data dictionary
        """
        with self._history_lock:
            # Add event to history
            event_index = len(self._history)
            self._history.append(event.copy())
            self._event_count += 1

            # Update type index
            event_type = event.get("type")
            if event_type:
                if event_type not in self._event_index:
                    self._event_index[event_type] = []
                self._event_index[event_type].append(event_index)

            # Update time index
            timestamp = event.get("timestamp")
            if timestamp is not None:
                self._time_index.append((timestamp, event_index))
                # Keep time index sorted
                self._time_index.sort(key=lambda x: x[0])

    def update_metric(self, name: str, value: int) -> None:
        """Update a metric value.

        Args:
            name: Metric name
            value: Value to add to metric
        """
        with self._metrics_lock:
            self._metrics[name] = self._metrics.get(name, 0) + value

    def get_metric(self, name: str) -> int:
        """Get a metric value.

        Args:
            name: Metric name

        Returns:
            The metric value

        Raises:
            KeyError: If metric does not exist
        """
        with self._metrics_lock:
            return self._metrics[name]

    def query_events(
        self, event_type: Optional[str] = None, start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Query events with optional filtering.

        Uses indexed lookups for efficient querying.

        Args:
            event_type: Optional event type filter
            start_time: Optional start time filter

        Returns:
            List of matching events
        """
        with self._history_lock:
            # Start with all indices
            matching_indices = set(range(len(self._history)))

            # Apply type filter if specified
            if event_type is not None:
                type_indices = set(self._event_index.get(event_type, []))
                matching_indices &= type_indices

            # Apply time filter if specified
            if start_time is not None:
                # Binary search for start time
                time_pos = self._binary_search_time(start_time)
                time_indices = {idx for _, idx in self._time_index[time_pos:]}
                matching_indices &= time_indices

            # Return events in chronological order
            return [self._history[i] for i in sorted(matching_indices)]

    def _binary_search_time(self, target_time: float) -> int:
        """Binary search for the first index >= target_time.

        Args:
            target_time: Target timestamp

        Returns:
            Index of first entry >= target_time
        """
        left, right = 0, len(self._time_index)
        while left < right:
            mid = (left + right) // 2
            if self._time_index[mid][0] < target_time:
                left = mid + 1
            else:
                right = mid
        return left

    def get_metrics_snapshot(self) -> Dict[str, int]:
        """Get a snapshot of all current metrics.

        Returns:
            Dictionary of metric name to value
        """
        with self._metrics_lock:
            return self._metrics.copy()

    def clear_history(self, before_time: Optional[float] = None) -> None:
        """Clear event history.

        Args:
            before_time: Optional timestamp to clear events before
        """
        with self._history_lock:
            if before_time is None:
                self._history.clear()
                self._event_index.clear()
                self._time_index.clear()
                self._event_count = 0
            else:
                # Find cutoff index
                cutoff = self._binary_search_time(before_time)
                remove_indices = {idx for _, idx in self._time_index[:cutoff]}

                # Update history
                new_history = [e for i, e in enumerate(self._history) if i not in remove_indices]
                self._history = new_history

                # Update indices
                self._event_index = {
                    t: [i for i in indices if i not in remove_indices] for t, indices in self._event_index.items()
                }
                self._time_index = self._time_index[cutoff:]
                self._event_count = len(self._history)
