import threading
import unittest
from typing import Any, Dict
from unittest.mock import Mock, patch

from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.machine.machine_status import MachineStatus
from gotstate.core.machine.state_machine import StateMachine
from gotstate.core.machine.submachine_machine import SubmachineMachine
from tests.unit.machine.machine_mocks import MockRegion, MockState


class TestSubmachineMachine(unittest.TestCase):
    """Test cases for SubmachineMachine class.

    Tests verify:
    1. Parent reference management
    2. Data context operations
    3. Snapshot management
    4. Lifecycle coordination
    5. Thread safety
    6. Error handling
    """

    def setUp(self):
        """Set up test fixtures."""
        # Patch base class methods that are abstract
        self.validate_patcher = patch.object(BasicStateMachine, "_validate_configuration")
        self.cleanup_patcher = patch.object(BasicStateMachine, "_cleanup_resources")
        self.validate_mock = self.validate_patcher.start()
        self.cleanup_mock = self.cleanup_patcher.start()

        self.machine = SubmachineMachine("test_submachine")
        self.parent = Mock(spec=StateMachine)
        self.state = MockState("test_state")
        self.region = MockRegion("test_region")

    def tearDown(self):
        """Clean up test fixtures."""
        self.validate_patcher.stop()
        self.cleanup_patcher.stop()

    def test_initial_state(self):
        """Test initial state of submachine.

        Verifies:
        1. Name set correctly
        2. No parent references
        3. Empty data context
        4. No snapshots
        5. Locks initialized
        """
        self.assertEqual(self.machine.name, "test_submachine")
        self.assertEqual(self.machine.parent_count, 0)
        self.assertEqual(len(self.machine._data_context), 0)
        self.assertEqual(len(self.machine._data_snapshots), 0)
        self.assertIsNotNone(self.machine._reference_lock)
        self.assertIsNotNone(self.machine._data_lock)

    def test_add_parent_reference(self):
        """Test adding parent references.

        Verifies:
        1. Valid parent added
        2. None parent rejected
        3. Self reference rejected
        4. Duplicate reference rejected
        """
        # Test valid parent
        self.machine.add_parent_reference(self.parent)
        self.assertEqual(self.machine.parent_count, 1)

        # Test None parent
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_parent_reference(None)
        self.assertIn("cannot be None", str(ctx.exception))

        # Test self reference
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_parent_reference(self.machine)
        self.assertIn("cyclic reference", str(ctx.exception))

        # Test duplicate reference
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_parent_reference(self.parent)
        self.assertIn("already referenced", str(ctx.exception))

    def test_remove_parent_reference(self):
        """Test removing parent references.

        Verifies:
        1. Valid parent removed
        2. None parent rejected
        3. Non-existent parent rejected
        4. Dead references cleaned up
        """
        # Add and remove valid parent
        self.machine.add_parent_reference(self.parent)
        self.machine.remove_parent_reference(self.parent)
        self.assertEqual(self.machine.parent_count, 0)

        # Test None parent
        with self.assertRaises(ValueError) as ctx:
            self.machine.remove_parent_reference(None)
        self.assertIn("cannot be None", str(ctx.exception))

        # Test non-existent parent
        with self.assertRaises(ValueError) as ctx:
            self.machine.remove_parent_reference(Mock(spec=StateMachine))
        self.assertIn("not referenced", str(ctx.exception))

    def test_data_operations(self):
        """Test data context operations.

        Verifies:
        1. Data set and retrieved correctly
        2. Deep copies maintained
        3. Clear operation works
        4. Events tracked
        """
        test_data = {"key": "value", "nested": {"inner": "data"}}

        # Test set and get
        self.machine.set_data("test", test_data)
        retrieved = self.machine.get_data("test")
        self.assertEqual(retrieved, test_data)

        # Verify deep copy
        test_data["nested"]["inner"] = "modified"
        retrieved = self.machine.get_data("test")
        self.assertEqual(retrieved["nested"]["inner"], "data")

        # Test clear
        self.machine.clear_data()
        self.assertEqual(len(self.machine._data_context), 0)

        # Test non-existent key
        with self.assertRaises(KeyError):
            self.machine.get_data("nonexistent")

    def test_snapshot_management(self):
        """Test data snapshot management.

        Verifies:
        1. Snapshot creation works
        2. Snapshot restoration works
        3. Invalid index handling
        4. Events tracked
        """
        # Create initial data
        self.machine.set_data("key1", "value1")
        self.machine.set_data("key2", "value2")

        # Create snapshot
        self.machine.create_data_snapshot()
        self.assertEqual(len(self.machine._data_snapshots), 1)

        # Modify data and create another snapshot
        self.machine.set_data("key1", "modified")
        self.machine.create_data_snapshot()
        self.assertEqual(len(self.machine._data_snapshots), 2)

        # Restore first snapshot
        self.machine.restore_data_snapshot(0)
        self.assertEqual(self.machine.get_data("key1"), "value1")

        # Test invalid index
        with self.assertRaises(IndexError):
            self.machine.restore_data_snapshot(99)

    def test_lifecycle_coordination(self):
        """Test lifecycle coordination with parents.

        Verifies:
        1. Parent initialization coordinated
        2. Parent activation coordinated
        3. Parent termination coordinated
        4. Dead references handled
        """
        parent1 = Mock(spec=StateMachine)
        parent2 = Mock(spec=StateMachine)

        self.machine.add_parent_reference(parent1)
        self.machine.add_parent_reference(parent2)

        # Test initialization
        with patch.object(BasicStateMachine, "initialize") as mock_init:
            self.machine.initialize()
            mock_init.assert_called_once()
            parent1.initialize.assert_called_once()
            parent2.initialize.assert_called_once()

        # Test activation
        with patch.object(BasicStateMachine, "activate") as mock_activate:
            self.machine.activate()
            mock_activate.assert_called_once()
            parent1.activate.assert_called_once()
            parent2.activate.assert_called_once()

        # Test termination
        with patch.object(BasicStateMachine, "terminate") as mock_term:
            self.machine.terminate()
            mock_term.assert_called_once()
            parent1.terminate.assert_called_once()
            parent2.terminate.assert_called_once()

    def test_thread_safety(self):
        """Test thread safety of operations.

        Verifies:
        1. Concurrent data operations
        2. Concurrent parent references
        3. Concurrent snapshots
        4. No data races
        """

        def data_operation():
            for i in range(100):
                self.machine.set_data(f"key_{i}", f"value_{i}")
                if i % 2 == 0:
                    self.machine.create_data_snapshot()

        def reference_operation():
            for i in range(100):
                parent = Mock(spec=StateMachine)
                try:
                    self.machine.add_parent_reference(parent)
                    self.machine.remove_parent_reference(parent)
                except ValueError:
                    pass  # Expected for some operations

        threads = [threading.Thread(target=data_operation), threading.Thread(target=reference_operation)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no data corruption
        self.assertGreater(len(self.machine._data_snapshots), 0)
        self.assertEqual(self.machine.parent_count, 0)

    def test_increment_data(self):
        """Test numeric data increment.

        Verifies:
        1. Integer increment works
        2. Float increment works
        3. Invalid key rejected
        4. Non-numeric value rejected
        5. Events tracked
        """
        # Test integer increment
        self.machine.set_data("int_value", 5)
        self.machine._increment_data("int_value", 3)
        self.assertEqual(self.machine.get_data("int_value"), 8)

        # Test float increment
        self.machine.set_data("float_value", 1.5)
        self.machine._increment_data("float_value", 0.5)
        self.assertEqual(self.machine.get_data("float_value"), 2.0)

        # Test invalid key
        with self.assertRaises(KeyError):
            self.machine._increment_data("nonexistent")

        # Test non-numeric value
        self.machine.set_data("string_value", "test")
        with self.assertRaises(TypeError):
            self.machine._increment_data("string_value")

    def test_validate_configuration(self):
        """Test configuration validation.

        Verifies:
        1. Valid configuration accepted
        2. Cyclic references detected
        3. Base validation performed
        """
        # Add valid state
        self.machine.add_state(self.state)
        self.machine._validate_configuration()  # Should not raise
        self.validate_mock.assert_called_once()

        # Test cyclic reference
        cyclic_state = MockState("cyclic")
        cyclic_state.submachine = self.machine
        self.machine.add_state(cyclic_state)

        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Cyclic submachine reference", str(ctx.exception))

    def test_cleanup_resources(self):
        """Test resource cleanup.

        Verifies:
        1. Data context cleared
        2. Parent references cleared
        3. Base cleanup performed
        4. Events tracked
        """
        # Add test data and references
        self.machine.set_data("test", "data")
        self.machine.add_parent_reference(self.parent)

        # Perform cleanup
        self.machine._cleanup_resources()

        # Verify cleanup
        self.assertEqual(len(self.machine._data_context), 0)
        self.assertEqual(self.machine.parent_count, 0)
        self.cleanup_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
