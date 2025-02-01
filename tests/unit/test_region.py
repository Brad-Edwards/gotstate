"""Unit tests for the Region class and its subclasses.

Tests the region management and parallel execution functionality.
"""

import logging
import threading
import time
import unittest
from unittest.mock import Mock, patch

import pytest

from gotstate.core.region import (
    HistoryRegion,
    ParallelRegion,
    Region,
    RegionManager,
    RegionStatus,
    SynchronizationRegion,
)
from gotstate.core.state import CompositeState, State, StateType


class TestRegion(unittest.TestCase):
    """Test cases for the Region class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent_state = CompositeState(state_id="parent")
        self.region = Region(region_id="test_region", parent_state=self.parent_state)
        self.state1 = State(state_id="state1", state_type=StateType.SIMPLE, parent=self.parent_state)
        self.state2 = State(state_id="state2", state_type=StateType.SIMPLE, parent=self.parent_state)

    def test_region_creation(self):
        """Test that a Region can be created with valid parameters."""
        self.assertEqual(self.region.id, "test_region")
        self.assertEqual(self.region.parent_state, self.parent_state)
        self.assertEqual(self.region.status, RegionStatus.INACTIVE)
        self.assertFalse(self.region.is_active)
        self.assertEqual(len(self.region.active_states), 0)

    def test_region_validation(self):
        """Test region validation rules."""
        # Test empty region ID
        with self.assertRaises(ValueError):
            Region(region_id="", parent_state=self.parent_state)

        # Test None region ID
        with self.assertRaises(ValueError):
            Region(region_id=None, parent_state=self.parent_state)

        # Test None parent state
        with self.assertRaises(ValueError):
            Region(region_id="test", parent_state=None)

    def test_region_activation(self):
        """Test region activation and deactivation."""
        # Test activation
        self.region.activate()
        self.assertEqual(self.region.status, RegionStatus.ACTIVE)
        self.assertTrue(self.region.is_active)

        # Test deactivation
        self.region.deactivate()
        self.assertEqual(self.region.status, RegionStatus.INACTIVE)
        self.assertFalse(self.region.is_active)

    def test_region_state_management(self):
        """Test active state management."""
        # Add active states
        self.region.add_active_state(self.state1)
        self.region.add_active_state(self.state2)

        # Verify active states
        active_states = self.region.active_states
        self.assertEqual(len(active_states), 2)
        self.assertIn(self.state1, active_states)
        self.assertIn(self.state2, active_states)

        # Remove active state
        self.region.remove_active_state(self.state1)
        active_states = self.region.active_states
        self.assertEqual(len(active_states), 1)
        self.assertNotIn(self.state1, active_states)
        self.assertIn(self.state2, active_states)

    def test_region_enter_exit(self):
        """Test region enter and exit behavior."""
        # Enter region
        self.region.enter()
        self.assertTrue(self.region.is_active)

        # Add some active states
        self.region.add_active_state(self.state1)
        self.region.add_active_state(self.state2)

        # Exit region
        self.region.exit()
        self.assertFalse(self.region.is_active)
        self.assertEqual(len(self.region.active_states), 0)

    def test_region_thread_safety(self):
        """Test thread safety of region operations."""
        import random
        import threading
        import time

        def worker():
            """Worker function for concurrent testing."""
            for _ in range(10):
                op = random.choice(["add", "remove", "activate", "deactivate"])
                if op == "add":
                    self.region.add_active_state(self.state1)
                elif op == "remove":
                    self.region.remove_active_state(self.state1)
                elif op == "activate":
                    self.region.activate()
                else:
                    self.region.deactivate()
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        # Create and start threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify region is in a consistent state
        self.assertIn(self.region.status, [RegionStatus.ACTIVE, RegionStatus.INACTIVE])

    def test_region_invalid_parent(self):
        """Test region creation with invalid parent state."""
        with pytest.raises(ValueError):
            Region("test_region", None)

    def test_region_invalid_id(self):
        """Test region creation with invalid ID."""
        mock_state = Mock()
        with pytest.raises(ValueError):
            Region("", mock_state)
        with pytest.raises(ValueError):
            Region(None, mock_state)


class TestParallelRegion(unittest.TestCase):
    """Test cases for the ParallelRegion class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent_state = CompositeState(state_id="parent")
        self.region = ParallelRegion(region_id="test_region", parent_state=self.parent_state)
        self.state1 = State(state_id="state1", state_type=StateType.SIMPLE, parent=self.parent_state)
        self.state2 = State(state_id="state2", state_type=StateType.SIMPLE, parent=self.parent_state)

    def tearDown(self):
        """Clean up after each test method."""
        self.region.deactivate()  # Ensure any running threads are stopped

    def test_parallel_region_creation(self):
        """Test that a ParallelRegion can be created with valid parameters."""
        self.assertEqual(self.region.id, "test_region")
        self.assertEqual(self.region.parent_state, self.parent_state)
        self.assertEqual(self.region.status, RegionStatus.INACTIVE)

    def test_parallel_region_execution(self):
        """Test parallel execution of region states."""
        # Mock the thread execution
        with patch.object(ParallelRegion, "_start_execution") as mock_start:
            # Add active states
            self.region.add_active_state(self.state1)
            self.region.add_active_state(self.state2)

            # Activate region
            self.region.activate()

            # Verify parallel execution
            self.assertTrue(self.region.is_active)
            active_states = self.region.active_states
            self.assertEqual(len(active_states), 2)
            self.assertIn(self.state1, active_states)
            self.assertIn(self.state2, active_states)

            # Verify thread was started
            mock_start.assert_called_once()

    def test_parallel_region_execution_error(self):
        """Test error handling during parallel region execution."""
        mock_state = Mock()
        region = ParallelRegion("test_region", mock_state)

        with patch.object(region, "_start_execution") as mock_start:
            mock_start.side_effect = RuntimeError("Execution failed")

            with self.assertRaises(RuntimeError):
                region.activate()

            self.assertEqual(region.status, RegionStatus.ACTIVE)

    def test_parallel_region_deactivation_during_execution(self):
        """Test parallel region deactivation while executing."""
        mock_state = Mock()
        region = ParallelRegion("test_region", mock_state)

        def mock_execute():
            while not region._stop_event.is_set():
                time.sleep(0.01)  # Shorter sleep and check stop event

        with patch.object(region, "_execute_region", side_effect=mock_execute):
            region.activate()
            time.sleep(0.02)  # Give thread time to start
            region.deactivate()  # Should stop cleanly now
            assert region.status == RegionStatus.INACTIVE
            assert region._execution_thread is None  # Verify thread cleanup


class TestSynchronizationRegion(unittest.TestCase):
    """Test cases for the SynchronizationRegion class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent_state = CompositeState(state_id="parent")
        self.region = SynchronizationRegion(region_id="test_region", parent_state=self.parent_state)
        self.state1 = State(state_id="state1", state_type=StateType.SIMPLE, parent=self.parent_state)
        self.state2 = State(state_id="state2", state_type=StateType.SIMPLE, parent=self.parent_state)

    def test_sync_point_management(self):
        """Test synchronization point management."""
        # Add sync points
        point1 = self.region.add_sync_point("join1", ["state1", "state2"])
        point2 = self.region.add_sync_point("fork1", ["state3", "state4"])

        # Verify sync points
        self.assertEqual(len(self.region.sync_points), 2)
        self.assertIn("join1", self.region.sync_points)
        self.assertIn("fork1", self.region.sync_points)

    def test_sync_point_completion(self):
        """Test synchronization point completion tracking."""
        point = self.region.add_sync_point("join1", ["state1", "state2"])

        # Mark states as complete
        self.region.mark_sync_complete("join1", "state1")
        self.assertFalse(point.is_complete)

        self.region.mark_sync_complete("join1", "state2")
        self.assertTrue(point.is_complete)

    def test_sync_point_validation(self):
        """Test synchronization point validation."""
        # Test empty point ID
        with self.assertRaises(ValueError):
            self.region.add_sync_point("", ["state1"])

        # Test empty participants
        with self.assertRaises(ValueError):
            self.region.add_sync_point("point1", [])

        # Test duplicate participants
        with self.assertRaises(ValueError):
            self.region.add_sync_point("point1", ["state1", "state1"])

    def test_sync_point_reset(self):
        """Test synchronization point reset."""
        point = self.region.add_sync_point("join1", ["state1", "state2"])

        # Complete and reset
        self.region.mark_sync_complete("join1", "state1")
        self.region.mark_sync_complete("join1", "state2")
        self.assertTrue(point.is_complete)

        point.reset()
        self.assertFalse(point.is_complete)

    def test_sync_point_invalid_participant(self):
        """Test marking completion for invalid participant."""
        mock_state = Mock()
        region = SynchronizationRegion("test_region", mock_state)
        sync_point = region.add_sync_point("test_point", ["state1", "state2"])

        # Try to mark invalid participant as complete
        region.mark_sync_complete("test_point", "invalid_state")
        assert not sync_point.is_complete

    def test_sync_point_duplicate_completion(self):
        """Test marking same participant complete multiple times."""
        mock_state = Mock()
        region = SynchronizationRegion("test_region", mock_state)
        sync_point = region.add_sync_point("test_point", ["state1", "state2"])

        region.mark_sync_complete("test_point", "state1")
        region.mark_sync_complete("test_point", "state1")  # Duplicate
        assert len(sync_point._completed) == 1


class TestHistoryRegion(unittest.TestCase):
    """Test cases for the HistoryRegion class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent_state = CompositeState(state_id="parent")
        self.region = HistoryRegion(region_id="test_region", parent_state=self.parent_state)
        self.state1 = State(state_id="state1", state_type=StateType.SIMPLE, parent=self.parent_state)
        self.state2 = State(state_id="state2", state_type=StateType.SIMPLE, parent=self.parent_state)

    def test_history_tracking(self):
        """Test history state tracking."""
        # Record state history
        self.region.record_state(self.state1)
        self.region.record_state(self.state2)

        # Verify history
        history = self.region.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1], self.state2)  # Most recent state

    def test_history_restoration(self):
        """Test history state restoration."""
        # Record and restore history
        self.region.record_state(self.state1)
        self.region.record_state(self.state2)

        restored_state = self.region.restore_history()
        self.assertEqual(restored_state, self.state2)  # Most recent state

    def test_deep_history(self):
        """Test deep history tracking and restoration."""
        # Create composite states with substates
        composite1 = CompositeState(state_id="comp1", parent=self.parent_state)
        substate1 = State(state_id="sub1", state_type=StateType.SIMPLE, parent=composite1)

        print("\nBefore recording:")
        print(f"composite1.id = {composite1.id}")
        print(f"substate1.id = {substate1.id}")

        # Record deep history
        self.region.record_state(composite1, deep=True)
        print("\nAfter record_state:")
        print(f"_history = {[s.id for s in self.region._history]}")
        print(f"_deep_history = {self.region._deep_history}")

        self.region.record_active_substate(composite1, substate1)
        print("\nAfter record_active_substate:")
        print(f"_history = {[s.id for s in self.region._history]}")
        print(f"_deep_history = {self.region._deep_history}")

        # Restore deep history
        restored_state, restored_substates = self.region.restore_deep_history()
        print("\nAfter restore_deep_history:")
        print(f"restored_state.id = {restored_state.id}")
        print(f"restored_substates = {restored_substates}")

        self.assertEqual(restored_state, composite1)
        self.assertEqual(restored_substates[composite1.id], substate1)

    def test_history_clearing(self):
        """Test history clearing."""
        # Record and clear history
        self.region.record_state(self.state1)
        self.region.record_state(self.state2)

        self.region.clear_history()
        history = self.region.get_history()
        self.assertEqual(len(history), 0)

    def test_history_state_recording_validation(self):
        """Test history state recording validation."""
        mock_state = Mock()
        mock_substate = Mock()
        region = HistoryRegion("test_region", mock_state)

        # Test with valid state
        mock_state.id = "test_state"
        region.record_state(mock_state)
        self.assertEqual(len(region._history), 1)

        # Test with None state (should raise ValueError)
        with self.assertRaises(ValueError):
            region.record_state(None)  # Implementation should reject None states
        self.assertEqual(len(region._history), 1)  # History length should not change

    def test_deep_history_restoration_empty(self):
        """Test deep history restoration when no history exists."""
        mock_state = Mock()
        region = HistoryRegion("test_region", mock_state)

        state, substates = region.restore_deep_history()
        assert state is None
        assert len(substates) == 0


class TestRegionManager(unittest.TestCase):
    """Test cases for the RegionManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.manager = RegionManager(max_regions=3)  # Small limit for testing
        self.parent_state = CompositeState(state_id="parent")
        self.region1 = Region(region_id="region1", parent_state=self.parent_state)
        self.region2 = Region(region_id="region2", parent_state=self.parent_state)
        self.region3 = Region(region_id="region3", parent_state=self.parent_state)

    def test_manager_creation(self):
        """Test that a RegionManager can be created with valid parameters."""
        self.assertEqual(self.manager._max_regions, 3)
        self.assertEqual(len(self.manager.active_regions), 0)

        # Test invalid max_regions
        with self.assertRaises(ValueError):
            RegionManager(max_regions=0)
        with self.assertRaises(ValueError):
            RegionManager(max_regions=-1)

    def test_region_management(self):
        """Test adding and removing regions."""
        # Add regions
        self.manager.add_region(self.region1)
        self.manager.add_region(self.region2)

        # Verify regions added
        active_regions = self.manager.active_regions
        self.assertEqual(len(active_regions), 2)
        self.assertIn(self.region1.id, active_regions)
        self.assertIn(self.region2.id, active_regions)

        # Test max regions limit
        self.manager.add_region(self.region3)
        with self.assertRaises(ValueError):
            self.manager.add_region(Region(region_id="region4", parent_state=self.parent_state))

        # Remove region
        self.manager.remove_region(self.region1.id)
        active_regions = self.manager.active_regions
        self.assertEqual(len(active_regions), 2)
        self.assertNotIn(self.region1.id, active_regions)

        # Test removing non-existent region
        with self.assertRaises(ValueError):
            self.manager.remove_region("non_existent")

    def test_dependency_management(self):
        """Test managing dependencies between regions."""
        # Add regions
        self.manager.add_region(self.region1)
        self.manager.add_region(self.region2)
        self.manager.add_region(self.region3)

        # Add dependencies
        self.manager.add_dependency(self.region2.id, self.region1.id)  # region2 depends on region1
        self.manager.add_dependency(self.region3.id, self.region2.id)  # region3 depends on region2

        # Test cycle detection
        with self.assertRaises(ValueError):
            self.manager.add_dependency(self.region1.id, self.region3.id)  # Would create cycle

        # Test self-dependency
        with self.assertRaises(ValueError):
            self.manager.add_dependency(self.region1.id, self.region1.id)

        # Remove dependency
        self.manager.remove_dependency(self.region2.id, self.region1.id)

        # Test removing non-existent dependency
        self.manager.remove_dependency(self.region1.id, self.region2.id)  # Should not raise error

    def test_activation_order(self):
        """Test region activation respects dependencies."""
        import logging

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        # Add regions and dependencies
        logger.debug("Adding regions")
        self.manager.add_region(self.region1)
        self.manager.add_region(self.region2)
        self.manager.add_region(self.region3)

        logger.debug("Adding dependencies")
        self.manager.add_dependency(self.region2.id, self.region1.id)
        self.manager.add_dependency(self.region3.id, self.region2.id)

        # Activate region3 (should activate 1 and 2 first)
        logger.debug("Activating region3")
        import threading

        activation_thread = threading.Thread(target=self.manager.activate_region, args=(self.region3.id,))
        activation_thread.start()
        activation_thread.join(timeout=5.0)  # Wait up to 5 seconds

        if activation_thread.is_alive():
            logger.error("Activation timed out!")
            self.manager.cleanup()  # Force cleanup
            self.fail("Region activation timed out after 5 seconds")

        # Verify all regions active
        logger.debug("Verifying activation")
        self.assertTrue(self.region1.is_active)
        self.assertTrue(self.region2.is_active)
        self.assertTrue(self.region3.is_active)

        # Deactivate region1 (should deactivate 2 and 3 first)
        logger.debug("Deactivating region1")
        deactivation_thread = threading.Thread(target=self.manager.deactivate_region, args=(self.region1.id,))
        deactivation_thread.start()
        deactivation_thread.join(timeout=5.0)  # Wait up to 5 seconds

        if deactivation_thread.is_alive():
            logger.error("Deactivation timed out!")
            self.manager.cleanup()  # Force cleanup
            self.fail("Region deactivation timed out after 5 seconds")

        # Verify all regions inactive
        logger.debug("Verifying deactivation")
        self.assertFalse(self.region1.is_active)
        self.assertFalse(self.region2.is_active)
        self.assertFalse(self.region3.is_active)

        logger.debug("Test completed successfully")

    def test_resource_management(self):
        """Test resource pool management."""
        # Add resources
        resource1 = "test_resource1"
        resource2 = {"key": "test_resource2"}

        self.manager.allocate_resource("res1", resource1)
        self.manager.allocate_resource("res2", resource2)

        # Get resources
        self.assertEqual(self.manager.get_resource("res1"), resource1)
        self.assertEqual(self.manager.get_resource("res2"), resource2)

        # Test duplicate resource
        with self.assertRaises(ValueError):
            self.manager.allocate_resource("res1", "duplicate")

        # Remove resource
        self.manager.deallocate_resource("res1")
        with self.assertRaises(ValueError):
            self.manager.get_resource("res1")

        # Test removing non-existent resource
        with self.assertRaises(ValueError):
            self.manager.deallocate_resource("non_existent")

    def test_cleanup(self):
        """Test cleanup of all resources and regions."""
        # Add regions, dependencies, and resources
        self.manager.add_region(self.region1)
        self.manager.add_region(self.region2)
        self.manager.add_dependency(self.region2.id, self.region1.id)
        self.manager.allocate_resource("res1", "test_resource")

        # Activate regions
        self.manager.activate_region(self.region2.id)

        # Cleanup
        self.manager.cleanup()

        # Verify everything is cleaned up
        self.assertEqual(len(self.manager.active_regions), 0)
        self.assertFalse(self.region1.is_active)
        self.assertFalse(self.region2.is_active)

        # Verify resources are cleaned up
        with self.assertRaises(ValueError):
            self.manager.get_resource("res1")

    def test_thread_safety(self):
        """Test thread safety of region manager operations."""
        import random
        import threading
        import time

        def worker():
            """Worker function for concurrent testing."""
            for _ in range(10):
                try:
                    op = random.choice(["add", "remove", "activate", "deactivate", "resource"])
                    if op == "add":
                        region = Region(f"region_{random.randint(0, 100)}", self.parent_state)
                        self.manager.add_region(region)
                    elif op == "remove":
                        regions = list(self.manager.active_regions.keys())
                        if regions:
                            self.manager.remove_region(random.choice(regions))
                    elif op == "activate":
                        regions = list(self.manager.active_regions.keys())
                        if regions:
                            self.manager.activate_region(random.choice(regions))
                    elif op == "deactivate":
                        regions = list(self.manager.active_regions.keys())
                        if regions:
                            self.manager.deactivate_region(random.choice(regions))
                    else:
                        self.manager.allocate_resource(
                            f"res_{random.randint(0, 100)}", f"resource_{random.randint(0, 100)}"
                        )
                except ValueError:
                    # Expected when hitting limits or duplicate IDs
                    pass
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        # Create and start threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Cleanup after thread test
        self.manager.cleanup()

    def test_region_manager_resource_handling(self):
        """Test region manager resource allocation and deallocation."""
        manager = RegionManager()

        # Allocate resource
        resource = "test_resource"
        manager.allocate_resource("res1", resource)
        self.assertEqual(manager.get_resource("res1"), resource)

        # Try to allocate duplicate
        with self.assertRaises(ValueError):
            manager.allocate_resource("res1", "another_resource")

        # Deallocate
        manager.deallocate_resource("res1")
        with self.assertRaises(ValueError):  # Changed from KeyError to ValueError
            manager.get_resource("res1")

    def test_region_manager_dependency_validation(self):
        """Test region manager dependency validation."""
        manager = RegionManager()
        mock_state = Mock()

        region1 = Region("region1", mock_state)
        region2 = Region("region2", mock_state)
        region3 = Region("region3", mock_state)

        manager.add_region(region1)
        manager.add_region(region2)
        manager.add_region(region3)

        # Add valid dependencies
        manager.add_dependency("region2", "region1")
        manager.add_dependency("region3", "region2")

        # Try to create cycle (should raise ValueError)
        with self.assertRaises(ValueError):
            manager.add_dependency("region1", "region3")

        # Remove dependency
        manager.remove_dependency("region2", "region1")

        # Remove non-existent dependency (should raise ValueError)
        with self.assertRaises(ValueError):
            manager.remove_dependency("region1", "non_existent")

    def test_region_manager_cleanup(self):
        """Test region manager cleanup."""
        manager = RegionManager()
        mock_state = Mock()

        # Add regions and resources
        region1 = Region("region1", mock_state)
        region2 = Region("region2", mock_state)
        manager.add_region(region1)
        manager.add_region(region2)
        manager.allocate_resource("res1", "resource1")

        # Add some dependencies
        manager.add_dependency("region2", "region1")

        # Cleanup
        manager.cleanup()

        # Verify everything is cleaned up
        self.assertEqual(len(manager.active_regions), 0)
        self.assertEqual(len(manager._resource_pool), 0)
        self.assertEqual(len(manager._dependencies), 0)


if __name__ == "__main__":
    unittest.main()
