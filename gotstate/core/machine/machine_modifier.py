import copy
import threading
from typing import Any, Dict

from gotstate.core.machine.machine_status import MachineStatus
from gotstate.core.machine.state_machine import StateMachine
from gotstate.core.machine.submachine_machine import SubmachineMachine


class MachineModifier:
    """Manages dynamic state machine modifications."""

    def __init__(self):
        """Initialize a new machine modifier."""
        self._machine = None
        self._prev_status = None
        self._snapshot = None
        self._modification_lock = threading.Lock()

    def _create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of machine state.

        Returns:
            Dictionary containing machine state snapshot
        """
        snapshot = {
            "states": copy.deepcopy(self._machine._states),
            "regions": copy.deepcopy(self._machine._regions),
            "transitions": copy.deepcopy(self._machine._transitions),
            "resources": copy.deepcopy(self._machine._resources),
            "status": self._machine.status,
        }

        if isinstance(self._machine, SubmachineMachine):
            snapshot["data_context"] = copy.deepcopy(self._machine._data_context)

        return snapshot

    def _restore_snapshot(self) -> None:
        """Restore machine state from snapshot."""
        if not self._snapshot:
            return

        self._machine._states = copy.deepcopy(self._snapshot["states"])
        self._machine._regions = copy.deepcopy(self._snapshot["regions"])
        self._machine._transitions = copy.deepcopy(self._snapshot["transitions"])
        self._machine._resources = copy.deepcopy(self._snapshot["resources"])

        if isinstance(self._machine, SubmachineMachine) and "data_context" in self._snapshot:
            self._machine._data_context = copy.deepcopy(self._snapshot["data_context"])

    def modify(self, machine: StateMachine):
        """Start a modification session.

        Args:
            machine: The machine to modify

        Returns:
            Context manager for the modification session

        Raises:
            ValueError: If machine is None
        """
        if machine is None:
            raise ValueError("Machine cannot be None")
        self._machine = machine
        return self

    def __enter__(self):
        """Enter the modification context.

        Sets machine status to MODIFYING and saves previous status.

        Raises:
            RuntimeError: If no machine specified or modification in progress
        """
        if not self._machine:
            raise RuntimeError("No machine specified for modification")

        if not self._modification_lock.acquire(blocking=False):
            raise RuntimeError("Another modification is in progress")

        try:
            with self._machine._status_lock:
                self._prev_status = self._machine.status
                self._machine._status = MachineStatus.MODIFYING
                self._snapshot = self._create_snapshot()
                self._machine._track_event("modification_started", {"prev_status": self._prev_status.name})
        except Exception as e:
            self._modification_lock.release()
            raise RuntimeError(f"Failed to start modification: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the modification context.

        Restores previous machine status unless an error occurred.
        """
        try:
            if not self._machine:
                return

            with self._machine._status_lock:
                if exc_type is not None:
                    # Error during modification, restore snapshot
                    self._restore_snapshot()
                    self._machine._track_event("modification_failed", {"error": str(exc_val)})
                else:
                    # Successful modification
                    self._machine._track_event("modification_completed", {"new_status": self._prev_status.name})

                self._machine._status = self._prev_status
        finally:
            # Clear state and release lock
            self._machine = None
            self._prev_status = None
            self._snapshot = None
            self._modification_lock.release()
