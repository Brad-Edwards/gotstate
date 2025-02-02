"""Unit tests for the state machine classes.

Tests the state machine orchestration and lifecycle management functionality.
"""

import threading
import time
import unittest
from typing import Any, Dict
from unittest.mock import Mock, patch
import weakref

from gotstate.core.event import Event, EventKind
from gotstate.core.machine import (
    BasicStateMachine,
    MachineBuilder,
    MachineModifier,
    MachineMonitor,
    MachineStatus,
    ProtocolMachine,
    StateMachine,
    SubmachineMachine,
)
from gotstate.core.region import Region
from gotstate.core.state import State, StateType
from gotstate.core.transition import Transition


class MockState(State):
    """Mock state for testing."""
    def __init__(self, state_id: str):
        """Initialize mock state."""
        super().__init__(state_id=state_id, state_type=StateType.SIMPLE)
        self.initialize = Mock()
        self.enter = Mock()
        self.exit = Mock()
        self.is_valid = Mock(return_value=True)

class MockRegion(Region):
    """Mock region for testing."""
    def __init__(self, region_id: str):
        """Initialize mock region."""
        mock_parent = MockState("mock_parent")
        super().__init__(region_id=region_id, parent_state=mock_parent)
        self.initialize = Mock()
        self.activate = Mock()
        self.deactivate = Mock()
        self.is_valid = Mock(return_value=True)

class TestBasicMachine(unittest.TestCase):
    """Test cases for the BasicStateMachine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.machine = BasicStateMachine()
        self.mock_state = MockState("test_state")
        self.mock_region = MockRegion("test_region")
        self.mock_transition = Mock(spec=Transition)
        self.mock_transition.is_valid = Mock(return_value=True)

    def test_machine_lifecycle(self):
        """Test machine lifecycle state transitions."""
        # Initial state
        self.assertEqual(self.machine.status, MachineStatus.UNINITIALIZED)

        # Initialize
        self.machine.initialize()
        self.assertEqual(self.machine.status, MachineStatus.INITIALIZING)

        # Activate
        self.machine.activate()
        self.assertEqual(self.machine.status, MachineStatus.ACTIVE)

        # Terminate
        self.machine.terminate()
        self.assertEqual(self.machine.status, MachineStatus.TERMINATED)

    def test_invalid_lifecycle_transitions(self):
        """Test invalid machine lifecycle state transitions."""
        # Cannot activate uninitialized machine
        with self.assertRaises(ValueError):
            self.machine.activate()

        # Cannot initialize twice
        self.machine.initialize()
        with self.assertRaises(ValueError):
            self.machine.initialize()

        # Cannot add components when not modifiable
        self.machine.activate()
        with self.assertRaises(ValueError):
            self.machine.add_state(self.mock_state)

    def test_component_management(self):
        """Test adding and managing machine components."""
        self.machine.initialize()

        # Add components
        self.machine.add_state(self.mock_state)
        self.machine.add_region(self.mock_region)
        self.machine.add_transition(self.mock_transition)

        # Verify components are initialized on activation
        self.machine.activate()
        self.mock_state.enter.assert_called_once()
        self.mock_region.activate.assert_called_once()

        # Verify components are cleaned up on termination
        self.machine.terminate()
        self.mock_state.exit.assert_called_once()
        self.mock_region.deactivate.assert_called_once()

    def test_event_processing(self):
        """Test event processing and handling."""
        self.machine.initialize()
        self.machine.activate()

        # Create and process an event
        mock_event = Mock(spec=Event)
        mock_event.kind = EventKind.SIGNAL
        self.machine.process_event(mock_event)

        # Verify event is processed
        # Note: Specific assertions depend on implementation

    def test_resource_management(self):
        """Test resource allocation and cleanup."""
        self.machine.initialize()

        # Add a resource
        mock_resource = Mock()
        self.machine.add_resource(mock_resource)

        # Verify resource is cleaned up on termination
        self.machine.terminate()
        mock_resource.cleanup.assert_called_once()

    def test_thread_safety(self):
        """Test thread-safe operations."""
        self.machine.initialize()
        self.machine.activate()

        def worker():
            """Worker function for concurrent testing."""
            mock_event = Mock(spec=Event)
            mock_event.kind = EventKind.SIGNAL
            self.machine.process_event(mock_event)

        # Create and start threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify machine remains in consistent state
        self.assertEqual(self.machine.status, MachineStatus.ACTIVE)

    def test_error_recovery_paths(self):
        """Test error recovery paths during machine operations."""
        machine = BasicStateMachine()

        # Mock components that will fail
        mock_state1 = MockState("state1")
        mock_state2 = MockState("state2")
        mock_state2.enter.side_effect = RuntimeError("Enter failed")

        mock_resource = Mock()
        mock_resource.id = "resource1"
        mock_resource.initialize = Mock()
        mock_resource.allocate = Mock()
        mock_resource.cleanup = Mock()
        mock_resource.is_valid = Mock(return_value=True)
        mock_resource.enter = Mock()
        mock_resource.exit = Mock()

        # Add components
        machine.add_state(mock_state1)
        machine.add_state(mock_state2)
        machine.add_resource(mock_resource)

        # Initialize should succeed
        machine.initialize()

        # Activation should fail due to mock_state2.enter failing
        with self.assertRaises(RuntimeError) as cm:
            machine.activate()
        self.assertIn("Enter failed", str(cm.exception))

        # Machine should be in INITIALIZING state after failed activation
        self.assertEqual(machine.status, MachineStatus.INITIALIZING)

    def test_initialization_component_failure(self):
        """Test error handling when component initialization fails."""
        machine = BasicStateMachine()
        mock_state = MockState("test_state")
        mock_state.initialize.side_effect = RuntimeError("Component init failed")

        machine.add_state(mock_state)

        with self.assertRaises(RuntimeError) as cm:
            machine.initialize()

        self.assertIn("Component init failed", str(cm.exception))
        self.assertEqual(machine.status, MachineStatus.UNINITIALIZED)
        mock_state.initialize.assert_called_once()

    def test_initialization_validation_failure(self):
        """Test error handling when configuration validation fails."""
        machine = BasicStateMachine()
        mock_state = MockState("test_state")
        mock_state.is_valid.return_value = False

        machine.add_state(mock_state)

        with self.assertRaises(ValueError) as cm:
            machine.initialize()

        self.assertIn("Invalid state configuration", str(cm.exception))
        self.assertEqual(machine.status, MachineStatus.UNINITIALIZED)
        mock_state.initialize.assert_called_once()
        mock_state.is_valid.assert_called_once()

    def test_activation_error_handling(self):
        """Test error handling during activation."""
        self.machine.add_state(self.mock_state)
        self.machine.initialize()

        # Test activation component error
        self.mock_state.enter.side_effect = ValueError("Test error")

        with self.assertRaises(RuntimeError) as cm:
            self.machine.activate()
        self.assertEqual(str(cm.exception), "Activation failed: Test error")
        self.assertEqual(self.machine.status, MachineStatus.INITIALIZING)

    def test_termination_error_handling(self):
        """Test error handling during termination."""
        self.machine.add_state(self.mock_state)
        self.machine.initialize()
        self.machine.activate()

        # Test termination component error
        self.mock_state.exit.side_effect = ValueError("Test error")

        with self.assertRaises(RuntimeError) as cm:
            self.machine.terminate()
        self.assertEqual(str(cm.exception), "Termination failed: Test error")
        self.assertEqual(self.machine.status, MachineStatus.ACTIVE)

    def test_validation_error_handling(self):
        """Test error handling during configuration validation."""
        # Test invalid state
        machine1 = BasicStateMachine()
        mock_state = MockState("test_state")
        mock_state.initialize = Mock()
        mock_state.enter = Mock()
        mock_state.exit = Mock()
        mock_state.is_valid = Mock(return_value=False)
        machine1.add_state(mock_state)

        with self.assertRaises(ValueError) as cm:
            machine1.initialize()
        self.assertIn("Invalid state configuration", str(cm.exception))

        # Test invalid region
        machine2 = BasicStateMachine()
        mock_region = MockRegion("test_region")
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=False)
        machine2.add_region(mock_region)

        with self.assertRaises(ValueError) as cm:
            machine2.initialize()
        self.assertIn("Invalid region configuration", str(cm.exception))

        # Test invalid transition
        machine3 = BasicStateMachine()
        mock_transition = Mock(spec=Transition)
        mock_transition.initialize = Mock()
        mock_transition.is_valid = Mock(return_value=False)
        machine3.add_transition(mock_transition)

        with self.assertRaises(ValueError) as cm:
            machine3.initialize()
        self.assertIn("Invalid transition configuration", str(cm.exception))

    def test_component_cleanup_on_error(self):
        """Test component cleanup when an error occurs during activation."""
        machine = BasicStateMachine()

        mock_state1 = MockState("state1")
        mock_state2 = MockState("state2")
        mock_state2.enter = Mock(side_effect=RuntimeError("Enter failed"))
        mock_state2.exit = Mock()
        mock_state2.is_valid = Mock(return_value=True)

        machine.add_state(mock_state1)
        machine.add_state(mock_state2)
        machine.initialize()

        with self.assertRaises(RuntimeError):
            machine.activate()

        # Verify first state was cleaned up
        mock_state1.exit.assert_called_once()
        self.assertEqual(machine.status, MachineStatus.INITIALIZING)

    def test_resource_allocation_failure(self):
        """Test error handling during resource allocation."""
        machine = BasicStateMachine()

        # Add a resource that will fail to allocate
        mock_resource = Mock()
        mock_resource.id = "test_resource"
        mock_resource.initialize = Mock()
        mock_resource.allocate = Mock(side_effect=RuntimeError("Allocation failed"))
        mock_resource.cleanup = Mock()
        mock_resource.is_valid = Mock(return_value=True)
        mock_resource.enter = Mock()
        mock_resource.exit = Mock()

        machine.add_resource(mock_resource)
        machine.initialize()

        with self.assertRaises(RuntimeError) as cm:
            machine.activate()

        self.assertIn("Allocation failed", str(cm.exception))
        mock_resource.allocate.assert_called_once()
        mock_resource.cleanup.assert_called_once()
        self.assertEqual(machine.status, MachineStatus.INITIALIZING)

    def test_state_transition_validation(self):
        """Test state transition validation paths."""
        machine = BasicStateMachine()

        # Create mock states with invalid transitions
        mock_state1 = MockState("state1")
        mock_state2 = MockState("state2")

        # Create an invalid transition
        mock_transition = Mock(spec=Transition)
        mock_transition.source = mock_state1
        mock_transition.target = mock_state2
        mock_transition.is_valid = Mock(return_value=False)

        # Add components
        machine.add_state(mock_state1)
        machine.add_state(mock_state2)
        machine.add_transition(mock_transition)

        # Initialize should fail due to invalid transition
        with self.assertRaises(ValueError) as cm:
            machine.initialize()

        self.assertIn("Invalid transition configuration", str(cm.exception))

    def test_version_management(self):
        """Test version management functionality."""
        # Test version getter
        self.assertEqual(self.machine.get_version(), "1.0.0")
        
        # Test version compatibility
        self.assertTrue(self.machine.validate_version_compatibility("1.1.0"))  # Compatible
        self.assertTrue(self.machine.validate_version_compatibility("1.2.0"))  # Compatible
        self.assertFalse(self.machine.validate_version_compatibility("2.0.0"))  # Incompatible
        self.assertFalse(self.machine.validate_version_compatibility("0.9.0"))  # Incompatible

    def test_security_policies(self):
        """Test security policy management."""
        # Add security policies
        def validate_read(op: str) -> bool:
            return op.startswith("read")
            
        def validate_write(op: str) -> bool:
            return op.startswith("write")
        
        # Add policies for read and write operations
        self.machine.add_security_policy("read_data", validate_read)
        self.machine.add_security_policy("write_data", validate_write)
        
        # Test policy validation
        self.assertTrue(self.machine.validate_security_policy("read_data"))  # Should pass - has policy and starts with "read"
        self.assertTrue(self.machine.validate_security_policy("write_data"))  # Should pass - has policy and starts with "write"
        self.assertFalse(self.machine.validate_security_policy("delete_data"))  # Should fail - no policy
        
        # Test operation without policy
        self.assertFalse(self.machine.validate_security_policy("unknown_operation"))  # No policy means denied (secure by default)

    def test_resource_limits(self):
        """Test resource limit management."""
        # Set custom limits
        self.machine.set_resource_limit("max_states", 2)
        self.machine.set_resource_limit("max_regions", 1)
        
        # Add components up to limits
        state1 = MockState("state1")
        state2 = MockState("state2")
        region = MockRegion("region1")
        
        self.machine.add_state(state1)
        self.machine.add_state(state2)
        self.machine.add_region(region)
        
        # Verify resource checks
        self.assertTrue(self.machine.check_resource_limits())
        
        # Exceed limits
        state3 = MockState("state3")
        with self.assertRaises(ValueError):
            self.machine.add_state(state3)
            self.machine.initialize()  # This should trigger resource limit check

    def test_component_validation_extended(self):
        """Test extended component validation scenarios."""
        # Test None component addition
        with self.assertRaises(ValueError):
            self.machine.add_state(None)
            
        with self.assertRaises(ValueError):
            self.machine.add_region(None)
            
        with self.assertRaises(ValueError):
            self.machine.add_resource(None)
            
        # Test duplicate IDs
        state1 = MockState("duplicate")
        state2 = MockState("duplicate")
        
        self.machine.add_state(state1)
        with self.assertRaises(ValueError):
            self.machine.add_state(state2)
            
        # Test invalid component types
        with self.assertRaises(ValueError):
            self.machine.add_state(Mock())  # Not a State
            
        with self.assertRaises(ValueError):
            self.machine.add_region(Mock())  # Not a Region


class TestProtocolMachine(unittest.TestCase):
    """Test cases for the ProtocolMachine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.machine = ProtocolMachine("test_protocol")

        # Create mock states
        self.state1 = MockState("state1")
        self.state2 = MockState("state2")
        self.initial_state = MockState("initial")
        self.running_state = MockState("running")
        self.stopped_state = MockState("stopped")

    def test_protocol_rule_management(self):
        """Test protocol rule addition and validation."""
        # Add states
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)

        # Add a valid rule
        rule = {
            "operation": "test_op",
            "source": "state1",
            "target": "state2",
        }
        self.machine.add_protocol_rule(rule)
        self.assertIn(rule, self.machine.protocol_rules)

        # Test invalid rule
        with self.assertRaises(ValueError):
            self.machine.add_protocol_rule({"invalid": "rule"})

    def test_protocol_operation_validation(self):
        """Test protocol operation validation."""
        # Add states
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)

        # Add rule
        rule = {
            "operation": "test_op",
            "source": "state1",
            "target": "state2",
        }
        self.machine.add_protocol_rule(rule)

        # Initialize and activate machine
        self.machine.initialize()
        self.machine.activate()

        # Set current state
        self.machine._current_state = self.state1

        # Test valid operation
        self.assertTrue(self.machine._validate_operation("test_op"))

        # Test invalid operation
        self.assertFalse(self.machine._validate_operation("invalid_op"))

    def test_protocol_sequence_validation(self):
        """Test protocol operation sequence validation."""
        # Add states
        self.machine.add_state(self.initial_state)
        self.machine.add_state(self.running_state)
        self.machine.add_state(self.stopped_state)

        # Add rules for a sequence
        self.machine.add_protocol_rule(
            {
                "operation": "start",
                "source": "initial",
                "target": "running",
            }
        )
        self.machine.add_protocol_rule(
            {
                "operation": "stop",
                "source": "running",
                "target": "stopped",
            }
        )

        # Initialize and activate machine
        self.machine.initialize()
        self.machine.activate()

        # Set initial state
        self.machine._current_state = self.initial_state

        # Test valid sequence
        self.assertTrue(self.machine._validate_operation("start"))
        self.machine._apply_operation("start")
        self.assertTrue(self.machine._validate_operation("stop"))

    def test_protocol_error_handling(self):
        """Test error handling in protocol operations."""
        # Add states
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)

        # Test invalid target state first
        rule = {
            "operation": "test_op",
            "source": "state1",
            "target": "nonexistent",
            "guard": Mock(return_value=True),
            "effect": Mock(),
        }
        self.machine.add_protocol_rule(rule)

        # Initialize and set current state
        self.machine.initialize()
        self.machine.activate()
        self.machine._current_state = self.state1

        # Test invalid target state
        with self.assertRaises(ValueError) as cm:
            self.machine._apply_operation("test_op")
        self.assertIn("Target state not found", str(cm.exception))

        # Create new machine for guard failure test
        machine2 = ProtocolMachine("test_protocol")
        machine2.add_state(self.state1)
        machine2.add_state(self.state2)
        rule = {
            "operation": "test_op",
            "source": "state1",
            "target": "state2",
            "guard": Mock(return_value=False),
            "effect": Mock(),
        }
        machine2.add_protocol_rule(rule)
        machine2.initialize()
        machine2.activate()
        machine2._current_state = self.state1

        with self.assertRaises(ValueError) as cm:
            machine2._apply_operation("test_op")
        self.assertIn("Guard condition failed", str(cm.exception))

        # Create new machine for effect error test
        machine3 = ProtocolMachine("test_protocol")
        machine3.add_state(self.state1)
        machine3.add_state(self.state2)
        rule = {
            "operation": "test_op",
            "source": "state1",
            "target": "state2",
            "guard": Mock(return_value=True),
            "effect": Mock(side_effect=ValueError("Effect error")),
        }
        machine3.add_protocol_rule(rule)
        machine3.initialize()
        machine3.activate()
        machine3._current_state = self.state1

        with self.assertRaises(ValueError) as cm:
            machine3._apply_operation("test_op")
        self.assertIn("Effect error", str(cm.exception))

    def test_protocol_event_handling(self):
        """Test protocol event handling."""
        # Add states
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)

        # Add rule
        self.machine.add_protocol_rule({"operation": "test_op", "source": "state1", "target": "state2"})

        # Initialize and set current state
        self.machine.initialize()
        self.machine.activate()
        self.machine._current_state = self.state1

        # Test non-call event
        event = Mock(spec=Event)
        event.kind = EventKind.SIGNAL
        self.machine.process_event(event)  # Should not raise

        # Test call event without operation
        event = Mock(spec=Event)
        event.kind = EventKind.CALL
        event.data = {}
        with self.assertRaises(ValueError) as cm:
            self.machine.process_event(event)
        self.assertIn("Call event must specify operation", str(cm.exception))

        # Test invalid operation sequence
        event.data = {"operation": "invalid_op"}
        with self.assertRaises(ValueError) as cm:
            self.machine.process_event(event)
        self.assertIn("Invalid operation", str(cm.exception))


class TestSubmachine(unittest.TestCase):
    """Test cases for the SubmachineMachine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.machine = SubmachineMachine("test_submachine")
        self.parent_machine = Mock(spec=StateMachine)

    def test_parent_reference_management(self):
        """Test parent machine reference management."""
        # Add parent reference
        self.machine.add_parent_reference(self.parent_machine)
        self.assertEqual(self.machine.parent_count, 1)

        # Remove parent reference
        self.machine.remove_parent_reference(self.parent_machine)
        self.assertEqual(self.machine.parent_count, 0)

        # Test cyclic reference prevention
        with self.assertRaises(ValueError):
            self.machine.add_parent_reference(self.machine)

    def test_data_context(self):
        """Test submachine data context management."""
        # Set data
        self.machine.set_data("key1", "value1")
        self.assertEqual(self.machine.get_data("key1"), "value1")

        # Clear data
        self.machine.clear_data()
        with self.assertRaises(KeyError):
            self.machine.get_data("key1")

    def test_lifecycle_coordination(self):
        """Test submachine lifecycle coordination with parent."""
        self.machine.add_parent_reference(self.parent_machine)

        # Initialize
        self.machine.initialize()
        self.parent_machine.initialize.assert_called_once()

        # Activate
        self.machine.activate()
        self.parent_machine.activate.assert_called_once()

        # Terminate
        self.machine.terminate()
        self.parent_machine.terminate.assert_called_once()

    def test_submachine_validation(self):
        """Test submachine configuration validation."""
        # Create a state with cyclic reference
        mock_state = MockState("cyclic_state")
        mock_state.initialize = Mock()
        mock_state.enter = Mock()
        mock_state.exit = Mock()
        mock_state.is_valid = Mock(return_value=True)

        # Create a submachine reference
        submachine_ref = weakref.ref(self.machine)
        mock_state.get_submachine = Mock(return_value=submachine_ref)
        mock_state.has_submachine = Mock(return_value=True)
        mock_state.submachine = self.machine  # Direct reference for validation

        # Add state and try to initialize
        self.machine.add_state(mock_state)

        # Add required region for initialization
        mock_region = MockRegion("test_region")
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=True)
        self.machine.add_region(mock_region)

        with self.assertRaises(ValueError) as cm:
            self.machine.initialize()
        self.assertIn("cyclic submachine reference detected", str(cm.exception).lower())

    def test_data_snapshot_management(self):
        """Test data snapshot creation and restoration."""
        machine = SubmachineMachine("test")
        
        # Set initial data
        machine.set_data("key1", "value1")
        machine.set_data("key2", [1, 2, 3])
        
        # Create snapshot
        machine.create_data_snapshot()
        
        # Modify data
        machine.set_data("key1", "modified")
        machine.set_data("key2", [4, 5, 6])
        
        # Create another snapshot
        machine.create_data_snapshot()
        
        # Restore first snapshot
        machine.restore_data_snapshot(0)
        self.assertEqual(machine.get_data("key1"), "value1")
        self.assertEqual(machine.get_data("key2"), [1, 2, 3])
        
        # Test invalid snapshot restoration
        with self.assertRaises(IndexError):
            machine.restore_data_snapshot(99)
            
        # Test restoring with no snapshots
        machine = SubmachineMachine("test2")
        with self.assertRaises(IndexError):
            machine.restore_data_snapshot()

    def test_data_isolation(self):
        """Test data context isolation."""
        machine = SubmachineMachine("test")
        
        # Test mutable data isolation
        original_list = [1, 2, 3]
        machine.set_data("list", original_list)
        
        # Modify original data
        original_list.append(4)
        stored_list = machine.get_data("list")
        self.assertEqual(stored_list, [1, 2, 3])  # Should not be affected
        
        # Modify retrieved data
        stored_list.append(5)
        self.assertEqual(machine.get_data("list"), [1, 2, 3])  # Should not be affected
        
        # Test nested data isolation
        nested_data = {"list": [1, 2], "dict": {"key": "value"}}
        machine.set_data("nested", nested_data)
        
        # Modify original nested data
        nested_data["list"].append(3)
        nested_data["dict"]["key"] = "modified"
        
        stored_nested = machine.get_data("nested")
        self.assertEqual(stored_nested["list"], [1, 2])
        self.assertEqual(stored_nested["dict"]["key"], "value")

    def test_data_operations_thread_safety(self):
        """Test thread safety of data operations."""
        machine = SubmachineMachine("test")
        machine.set_data("counter", 0)
        
        def worker():
            for _ in range(100):
                machine._increment_data("counter")
                
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        self.assertEqual(machine.get_data("counter"), 1000)  # 10 threads * 100 increments

    def test_data_increment_validation(self):
        """Test data increment validation."""
        machine = SubmachineMachine("test")
        
        # Test increment of non-existent key
        with self.assertRaises(KeyError):
            machine._increment_data("non_existent")
            
        # Test increment of non-numeric value
        machine.set_data("string", "value")
        with self.assertRaises(TypeError):
            machine._increment_data("string")
            
        # Test valid increment
        machine.set_data("number", 5)
        machine._increment_data("number")
        self.assertEqual(machine.get_data("number"), 6)
        
        # Test custom increment amount
        machine._increment_data("number", 10)
        self.assertEqual(machine.get_data("number"), 16)


class TestMachineMonitor(unittest.TestCase):
    """Test cases for the MachineMonitor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.monitor = MachineMonitor()

    def test_event_tracking(self):
        """Test event tracking and querying."""
        # Track events
        event1 = {"type": "state_change", "time": 100}
        event2 = {"type": "transition", "time": 200}
        self.monitor.track_event(event1)
        self.monitor.track_event(event2)

        # Query events
        events = self.monitor.query_events("state_change")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], event1)

    def test_metrics(self):
        """Test metric tracking and querying."""
        # Update metrics
        self.monitor.update_metric("events_processed", 10)
        self.monitor.update_metric("transitions_fired", 5)

        # Get metrics
        self.assertEqual(self.monitor.get_metric("events_processed"), 10)
        self.assertEqual(self.monitor.get_metric("transitions_fired"), 5)

    def test_thread_safety(self):
        """Test thread-safe monitoring operations."""

        def worker():
            """Worker function for concurrent testing."""
            for i in range(100):
                self.monitor.update_metric("counter", i)
                self.monitor.track_event({"type": "test", "value": i})

        # Create and start threads
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify monitor state is consistent
        self.assertEqual(len(self.monitor.history), 1000)  # 10 threads * 100 events

    def test_event_querying(self):
        """Test event querying functionality."""
        monitor = MachineMonitor()
        
        # Add events with different types and timestamps
        events = [
            {"type": "state_change", "timestamp": 1000.0, "data": "event1"},
            {"type": "transition", "timestamp": 1001.0, "data": "event2"},
            {"type": "state_change", "timestamp": 1002.0, "data": "event3"},
            {"type": "error", "timestamp": 1003.0, "data": "event4"}
        ]
        
        for event in events:
            monitor.track_event(event)
            
        # Query by type
        state_changes = monitor.query_events(event_type="state_change")
        self.assertEqual(len(state_changes), 2)
        self.assertEqual(state_changes[0]["data"], "event1")
        
        # Query by time
        recent_events = monitor.query_events(start_time=1002.0)
        self.assertEqual(len(recent_events), 2)
        self.assertEqual(recent_events[0]["data"], "event3")
        
        # Query with both filters
        filtered_events = monitor.query_events(event_type="state_change", start_time=1002.0)
        self.assertEqual(len(filtered_events), 1)
        self.assertEqual(filtered_events[0]["data"], "event3")

    def test_history_management(self):
        """Test history management functionality."""
        monitor = MachineMonitor()
        
        # Add events
        for i in range(5):
            monitor.track_event({
                "type": "test",
                "timestamp": 1000.0 + i,
                "data": f"event{i}"
            })
            
        # Clear history before specific time
        monitor.clear_history(before_time=1002.0)
        
        # Verify remaining events
        history = monitor.history
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["data"], "event2")
        
        # Clear all history
        monitor.clear_history()
        self.assertEqual(len(monitor.history), 0)
        self.assertEqual(monitor.event_count, 0)

    def test_metrics_management(self):
        """Test metrics management functionality."""
        monitor = MachineMonitor()
        
        # Update metrics
        monitor.update_metric("counter1", 5)
        monitor.update_metric("counter2", 3)
        monitor.update_metric("counter1", 2)  # Increment existing
        
        # Get individual metrics
        self.assertEqual(monitor.get_metric("counter1"), 7)
        self.assertEqual(monitor.get_metric("counter2"), 3)
        
        # Get metrics snapshot
        metrics = monitor.get_metrics_snapshot()
        self.assertEqual(metrics["counter1"], 7)
        self.assertEqual(metrics["counter2"], 3)
        
        # Test non-existent metric
        with self.assertRaises(KeyError):
            monitor.get_metric("non_existent")

    def test_event_index_integrity(self):
        """Test integrity of event indexing."""
        monitor = MachineMonitor()
        
        # Add events with gaps in timestamps
        events = [
            {"type": "type1", "timestamp": 1000.0},
            {"type": "type2", "timestamp": 1005.0},
            {"type": "type1", "timestamp": 1010.0},
            {"type": "type3", "timestamp": 1015.0}
        ]
        
        for event in events:
            monitor.track_event(event)
            
        # Verify type index
        type1_events = monitor.query_events(event_type="type1")
        self.assertEqual(len(type1_events), 2)
        self.assertEqual(type1_events[0]["timestamp"], 1000.0)
        self.assertEqual(type1_events[1]["timestamp"], 1010.0)
        
        # Verify time index with gaps
        mid_events = monitor.query_events(start_time=1003.0)
        self.assertEqual(len(mid_events), 3)
        self.assertEqual(mid_events[0]["timestamp"], 1005.0)
