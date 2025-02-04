import copy
import threading
import weakref
from typing import Any, Dict, List, Set

from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.machine.state_machine import StateMachine


class SubmachineMachine(BasicStateMachine):
    """Represents a submachine state machine.

    SubmachineMachine implements reusable state machine components
    that can be referenced by other machines.

    Class Invariants:
    1. Must maintain encapsulation
    2. Must handle references
    3. Must coordinate lifecycle
    4. Must isolate data

    Design Patterns:
    - Proxy: Manages references
    - Flyweight: Shares instances
    - Bridge: Decouples interface

    Threading/Concurrency Guarantees:
    1. Thread-safe reference management
    2. Atomic lifecycle operations
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) reference management
    2. O(r) coordination where r is reference count
    3. O(d) data isolation where d is data size
    """

    def __init__(self, name: str) -> None:
        """Initialize the submachine.

        Args:
            name: The name of the submachine
        """
        super().__init__()
        self._name = name
        self._parent_machines: Set[weakref.ReferenceType[StateMachine]] = set()
        self._reference_lock = threading.Lock()
        self._data_context: Dict[str, Any] = {}
        self._data_lock = threading.Lock()
        self._data_snapshots: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Get the submachine name."""
        return self._name

    @property
    def parent_count(self) -> int:
        """Get the number of parent machines.

        Returns:
            The number of parent machines
        """
        with self._reference_lock:
            # Clean up dead references first
            self._parent_machines = {ref for ref in self._parent_machines if ref() is not None}
            return len(self._parent_machines)

    def add_parent_reference(self, parent: StateMachine) -> None:
        """Add a parent machine reference.

        Args:
            parent: The parent machine to reference

        Raises:
            ValueError: If parent is None, already referenced, or would create a cycle
        """
        if parent is None:
            raise ValueError("Parent machine cannot be None")

        if parent is self:
            raise ValueError("Cannot add self as parent (cyclic reference)")

        with self._reference_lock:
            # Clean up dead references
            self._parent_machines = {ref for ref in self._parent_machines if ref() is not None}

            # Check for existing reference
            for ref in self._parent_machines:
                if ref() is parent:
                    raise ValueError("Parent machine already referenced")

            # Add new reference using weakref.ref
            self._parent_machines.add(weakref.ref(parent))

    def remove_parent_reference(self, parent: StateMachine) -> None:
        """Remove a parent machine reference.

        Args:
            parent: The parent machine to remove

        Raises:
            ValueError: If parent is None or not referenced
        """
        if parent is None:
            raise ValueError("Parent machine cannot be None")

        with self._reference_lock:
            # Clean up dead references first
            self._parent_machines = {ref for ref in self._parent_machines if ref() is not None}

            # Find and remove the reference
            to_remove = None
            for ref in self._parent_machines:
                if ref() is parent:
                    to_remove = ref
                    break

            if to_remove is None:
                raise ValueError("Parent machine not referenced")

            self._parent_machines.remove(to_remove)

    def get_data(self, key: str) -> Any:
        """Get data from the submachine context.

        Args:
            key: The data key

        Returns:
            The data value (deep copy)

        Raises:
            KeyError: If key not found
        """
        with self._data_lock:
            if key not in self._data_context:
                raise KeyError(f"Data key not found: {key}")
            return copy.deepcopy(self._data_context[key])

    def set_data(self, key: str, value: Any) -> None:
        """Set data in the submachine context.

        Args:
            key: The data key
            value: The data value
        """
        with self._data_lock:
            # Store a deep copy to ensure isolation
            self._data_context[key] = copy.deepcopy(value)
            self._track_event("data_modified", {"key": key, "operation": "set"})

    def clear_data(self) -> None:
        """Clear all data from the submachine context."""
        with self._data_lock:
            self._data_context.clear()
            self._track_event("data_cleared", {})

    def create_data_snapshot(self) -> None:
        """Create a snapshot of current data context."""
        with self._data_lock:
            snapshot = copy.deepcopy(self._data_context)
            self._data_snapshots.append(snapshot)
            self._track_event("data_snapshot_created", {"snapshot_id": len(self._data_snapshots) - 1})

    def restore_data_snapshot(self, index: int = -1) -> None:
        """Restore data from a snapshot.

        Args:
            index: Index of snapshot to restore (-1 for most recent)

        Raises:
            IndexError: If snapshot index is invalid
        """
        with self._data_lock:
            if not self._data_snapshots:
                raise IndexError("No snapshots available")
            if index < -len(self._data_snapshots) or index >= len(self._data_snapshots):
                raise IndexError("Invalid snapshot index")

            snapshot = self._data_snapshots[index]
            self._data_context = copy.deepcopy(snapshot)
            self._track_event("data_snapshot_restored", {"snapshot_id": index})

    def initialize(self) -> None:
        """Initialize the submachine and coordinate with parents."""
        super().initialize()

        # Initialize parents first
        with self._reference_lock:
            for ref in self._parent_machines:
                parent = ref()
                if parent is not None:
                    parent.initialize()

    def activate(self) -> None:
        """Activate the submachine and coordinate with parents."""
        super().activate()

        # Activate parents
        with self._reference_lock:
            for ref in self._parent_machines:
                parent = ref()
                if parent is not None:
                    parent.activate()

    def terminate(self) -> None:
        """Terminate the submachine and coordinate with parents."""
        # Terminate parents first
        with self._reference_lock:
            for ref in self._parent_machines:
                parent = ref()
                if parent is not None:
                    parent.terminate()

        super().terminate()

        # Clear data context on termination
        self.clear_data()

    def _validate_configuration(self) -> None:
        """Validate submachine configuration.

        Raises:
            ValueError: If cyclic reference detected
        """
        # Validate base configuration first
        super()._validate_configuration()

        # Additional validation for submachine
        with self._collection_lock:
            # Ensure no cyclic references in states
            for state in self._states.values():
                if hasattr(state, "submachine"):
                    submachine = getattr(state, "submachine")
                    if submachine is self:
                        raise ValueError("Cyclic submachine reference detected")

    def _cleanup_resources(self) -> None:
        """Clean up submachine resources."""
        # Clear data context
        with self._data_lock:
            self._data_context.clear()

        # Clear parent references
        with self._reference_lock:
            self._parent_machines.clear()

        super()._cleanup_resources()

    def _increment_data(self, key: str, increment: int = 1) -> None:
        """Thread-safe increment of numeric data.

        Args:
            key: The data key
            increment: Amount to increment by

        Raises:
            KeyError: If key not found
            TypeError: If value is not numeric
        """
        with self._data_lock:
            if key not in self._data_context:
                raise KeyError(f"Data key not found: {key}")
            value = self._data_context[key]
            if not isinstance(value, (int, float)):
                raise TypeError(f"Value for key '{key}' is not numeric")
            self._data_context[key] = value + increment
            self._track_event("data_modified", {"key": key, "operation": "increment", "increment": increment})
