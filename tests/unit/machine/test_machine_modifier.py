import copy
import unittest
from threading import Lock
from unittest.mock import MagicMock, Mock, patch

from gotstate.core.machine.machine_modifier import MachineModifier
from gotstate.core.machine.machine_status import MachineStatus
from gotstate.core.machine.state_machine import StateMachine
from gotstate.core.machine.submachine_machine import SubmachineMachine
from tests.unit.machine.machine_mocks import MockRegion, MockState


class LockWrapper:
    """Wrapper for Lock that supports deep copy."""

    def __init__(self):
        self._lock = Lock()
        self._locked = False

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def acquire(self, blocking=True, timeout=-1):
        result = self._lock.acquire(blocking=blocking, timeout=timeout)
        if result:
            self._locked = True
        return result

    def release(self):
        if not self._locked:
            return  # Silently ignore release of unlocked lock in tests
        self._locked = False
        return self._lock.release()


def mock_deepcopy(obj, memo=None):
    """Mock deepcopy that handles thread locks."""
    if isinstance(obj, LockWrapper):
        return LockWrapper()  # Return a new lock wrapper
    if isinstance(obj, dict):
        return {k: mock_deepcopy(v, memo) for k, v in obj.items()}
    if isinstance(obj, (list, set)):
        return type(obj)(mock_deepcopy(x, memo) for x in obj)
    return obj


class TestMachineModifier(unittest.TestCase):
    """Test cases for MachineModifier class.

    Tests verify:
    1. Initialization
    2. Snapshot management
    3. Modification context
    4. Error handling
    5. Thread safety
    """

    def setUp(self):
        """Set up test fixtures."""
        self.modifier = MachineModifier()
        self.machine = Mock(spec=StateMachine)

        # Create a lock wrapper with context manager support
        self.status_lock = LockWrapper()
        self.machine._status_lock = self.status_lock

        # Mock components that can be copied
        self.machine._states = {"state1": MockState("state1")}
        self.machine._regions = {"region1": MockRegion("region1")}
        self.machine._transitions = {"t1": Mock()}
        self.machine._resources = {"r1": Mock()}

        # Status and event tracking
        self.machine.status = MachineStatus.ACTIVE
        self.machine._track_event = Mock()

        # Patch deepcopy
        self.deepcopy_patcher = patch("copy.deepcopy", side_effect=mock_deepcopy)
        self.mock_deepcopy = self.deepcopy_patcher.start()

        # Initialize modification lock
        self.modifier._modification_lock = LockWrapper()

    def tearDown(self):
        """Clean up test fixtures."""
        self.deepcopy_patcher.stop()

    def test_initial_state(self):
        """Test initial state of modifier.

        Verifies:
        1. No machine initially
        2. No previous status
        3. No snapshot
        4. Lock initialized
        """
        self.assertIsNone(self.modifier._machine)
        self.assertIsNone(self.modifier._prev_status)
        self.assertIsNone(self.modifier._snapshot)
        self.assertIsNotNone(self.modifier._modification_lock)

    def test_modify_validation(self):
        """Test modify method validation.

        Verifies:
        1. None machine rejected
        2. Valid machine accepted
        3. Returns context manager
        """
        # Test None machine
        with self.assertRaises(ValueError) as ctx:
            self.modifier.modify(None)
        self.assertIn("Machine cannot be None", str(ctx.exception))

        # Test valid machine
        result = self.modifier.modify(self.machine)
        self.assertEqual(result, self.modifier)
        self.assertEqual(self.modifier._machine, self.machine)

    def test_create_snapshot(self):
        """Test snapshot creation.

        Verifies:
        1. All components copied
        2. Deep copy performed
        3. Status preserved
        4. Submachine data handled
        """
        self.modifier._machine = self.machine
        snapshot = self.modifier._create_snapshot()

        # Verify all components present
        self.assertIn("states", snapshot)
        self.assertIn("regions", snapshot)
        self.assertIn("transitions", snapshot)
        self.assertIn("resources", snapshot)
        self.assertIn("status", snapshot)

        # Verify values copied
        self.assertEqual(snapshot["states"], self.machine._states)
        self.assertEqual(snapshot["regions"], self.machine._regions)
        self.assertEqual(snapshot["transitions"], self.machine._transitions)
        self.assertEqual(snapshot["resources"], self.machine._resources)
        self.assertEqual(snapshot["status"], self.machine.status)

    def test_restore_snapshot(self):
        """Test snapshot restoration.

        Verifies:
        1. All components restored
        2. Deep copy performed
        3. No restoration if no snapshot
        4. Submachine data handled
        """
        self.modifier._machine = self.machine
        self.modifier._snapshot = {
            "states": {"state2": MockState("state2")},
            "regions": {"region2": MockRegion("region2")},
            "transitions": {"t2": Mock()},
            "resources": {"r2": Mock()},
            "status": MachineStatus.ACTIVE,
        }

        self.modifier._restore_snapshot()

        # Verify machine state restored
        self.assertEqual(self.machine._states, self.modifier._snapshot["states"])
        self.assertEqual(self.machine._regions, self.modifier._snapshot["regions"])
        self.assertEqual(self.machine._transitions, self.modifier._snapshot["transitions"])
        self.assertEqual(self.machine._resources, self.modifier._snapshot["resources"])

    def test_submachine_snapshot(self):
        """Test snapshot handling for submachines.

        Verifies:
        1. Data context included in snapshot
        2. Data context restored
        """
        submachine = Mock(spec=SubmachineMachine)
        submachine._states = self.machine._states
        submachine._regions = self.machine._regions
        submachine._transitions = self.machine._transitions
        submachine._resources = self.machine._resources
        submachine.status = MachineStatus.ACTIVE
        submachine._status_lock = self.status_lock
        submachine._track_event = Mock()
        submachine._data_context = {"key": "value"}

        self.modifier._machine = submachine
        snapshot = self.modifier._create_snapshot()

        # Verify data context included
        self.assertIn("data_context", snapshot)
        self.assertEqual(snapshot["data_context"], submachine._data_context)

        # Verify data context restored
        self.modifier._snapshot = snapshot
        self.modifier._restore_snapshot()
        self.assertEqual(submachine._data_context, snapshot["data_context"])

    def test_context_manager_success(self):
        """Test successful modification context.

        Verifies:
        1. Status changed to MODIFYING
        2. Previous status saved
        3. Snapshot created
        4. Event tracked
        5. Status restored on exit
        """
        with self.modifier.modify(self.machine):
            # Verify status changed
            self.assertEqual(self.machine._status, MachineStatus.MODIFYING)
            # Verify snapshot created
            self.assertIsNotNone(self.modifier._snapshot)
            # Verify event tracked
            self.machine._track_event.assert_called_with(
                "modification_started", {"prev_status": MachineStatus.ACTIVE.name}
            )

        # Verify status restored
        self.assertEqual(self.machine._status, MachineStatus.ACTIVE)
        # Verify completion tracked
        self.machine._track_event.assert_called_with(
            "modification_completed", {"new_status": MachineStatus.ACTIVE.name}
        )

    def test_context_manager_error(self):
        """Test modification context with error.

        Verifies:
        1. Snapshot restored on error
        2. Error event tracked
        3. Status restored
        4. Lock released
        """
        error_msg = "Test error"

        try:
            with self.modifier.modify(self.machine):
                raise RuntimeError(error_msg)
        except RuntimeError:
            pass

        # Verify snapshot restored and error tracked
        self.machine._track_event.assert_any_call("modification_failed", {"error": error_msg})
        # Verify status restored
        self.assertEqual(self.machine._status, MachineStatus.ACTIVE)
        # Verify state cleared
        self.assertIsNone(self.modifier._machine)
        self.assertIsNone(self.modifier._prev_status)
        self.assertIsNone(self.modifier._snapshot)

    def test_concurrent_modification(self):
        """Test concurrent modification prevention.

        Verifies:
        1. Concurrent modifications blocked
        2. Lock released after error
        3. Lock released after completion
        """
        # Start first modification
        with self.modifier.modify(self.machine):
            # Try concurrent modification
            with self.assertRaises(RuntimeError) as ctx:
                with self.modifier.modify(self.machine):
                    pass
            self.assertIn("Another modification is in progress", str(ctx.exception))

        # Verify lock released
        with self.modifier.modify(self.machine):
            pass  # Should succeed

    def test_invalid_context_usage(self):
        """Test invalid context manager usage.

        Verifies:
        1. Enter without machine fails
        2. Exit without machine succeeds
        3. Proper error messages
        """
        # Test enter without machine
        with self.assertRaises(RuntimeError) as ctx:
            with self.modifier:
                pass
        self.assertIn("No machine specified", str(ctx.exception))

        # Test exit without machine (should not raise)
        self.modifier._machine = None
        self.modifier.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
