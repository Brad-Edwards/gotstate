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


class TestBasicMachine(unittest.TestCase):
    """Test cases for the BasicStateMachine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.machine = BasicStateMachine()
        self.mock_state = Mock(spec=State)
        self.mock_state.id = "test_state"
        self.mock_state.initialize = Mock()
        self.mock_state.enter = Mock()
        self.mock_state.exit = Mock()
        self.mock_state.is_valid = Mock(return_value=True)

        self.mock_region = Mock(spec=Region)
        self.mock_region.id = "test_region"
        self.mock_region.initialize = Mock()
        self.mock_region.activate = Mock()
        self.mock_region.deactivate = Mock()
        self.mock_region.is_valid = Mock(return_value=True)

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

    def test_initialization_component_failure(self):
        """Test error handling when component initialization fails."""
        machine = BasicStateMachine()
        mock_state = Mock()
        mock_state.id = "test_state"
        mock_state.initialize = Mock(side_effect=RuntimeError("Component init failed"))
        mock_state.is_valid = Mock(return_value=True)
        mock_state.enter = Mock()
        mock_state.exit = Mock()

        machine.add_state(mock_state)

        with self.assertRaises(RuntimeError) as cm:
            machine.initialize()

        self.assertIn("Component init failed", str(cm.exception))
        self.assertEqual(machine.status, MachineStatus.UNINITIALIZED)
        mock_state.initialize.assert_called_once()

    def test_initialization_validation_failure(self):
        """Test error handling when configuration validation fails."""
        machine = BasicStateMachine()
        mock_state = Mock()
        mock_state.id = "test_state"
        mock_state.initialize = Mock()
        mock_state.is_valid = Mock(return_value=False)
        mock_state.enter = Mock()
        mock_state.exit = Mock()

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
        mock_state = Mock(spec=State)
        mock_state.id = "test_state"
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
        mock_region = Mock(spec=Region)
        mock_region.id = "test_region"
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

        mock_state1 = Mock(spec=State)
        mock_state1.id = "state1"
        mock_state1.initialize = Mock()
        mock_state1.enter = Mock()
        mock_state1.exit = Mock()
        mock_state1.is_valid = Mock(return_value=True)

        mock_state2 = Mock(spec=State)
        mock_state2.id = "state2"
        mock_state2.initialize = Mock()
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

    def test_error_recovery_paths(self):
        """Test error recovery paths during machine operations."""
        machine = BasicStateMachine()

        # Mock components that will fail
        mock_state1 = Mock()
        mock_state1.id = "state1"
        mock_state1.initialize = Mock()
        mock_state1.enter = Mock()
        mock_state1.exit = Mock()
        mock_state1.is_valid = Mock(return_value=True)

        mock_state2 = Mock()
        mock_state2.id = "state2"
        mock_state2.initialize = Mock()
        mock_state2.enter = Mock(side_effect=RuntimeError("Enter failed"))
        mock_state2.exit = Mock()
        mock_state2.is_valid = Mock(return_value=True)

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
        self.assertEqual(machine.status, MachineStatus.INITIALIZING)

        # Activation should fail and trigger cleanup
        with self.assertRaises(RuntimeError):
            machine.activate()

        # Verify cleanup occurred
        mock_state1.exit.assert_called_once()
        mock_resource.cleanup.assert_called_once()
        self.assertEqual(machine.status, MachineStatus.INITIALIZING)

        # Termination should still work
        machine.terminate()
        self.assertEqual(machine.status, MachineStatus.TERMINATED)

    def test_state_transition_validation(self):
        """Test state transition validation paths."""
        machine = BasicStateMachine()

        # Create mock states with invalid transitions
        mock_state1 = Mock()
        mock_state1.id = "state1"
        mock_state1.initialize = Mock()
        mock_state1.enter = Mock()
        mock_state1.exit = Mock()
        mock_state1.is_valid = Mock(return_value=True)

        mock_state2 = Mock()
        mock_state2.id = "state2"
        mock_state2.initialize = Mock()
        mock_state2.enter = Mock()
        mock_state2.exit = Mock()
        mock_state2.is_valid = Mock(return_value=True)

        # Create an invalid transition
        mock_transition = Mock()
        mock_transition.source = mock_state1
        mock_transition.target = mock_state2
        mock_transition.is_valid = Mock(return_value=False)

        # Add components
        machine.add_state(mock_state1)
        machine.add_state(mock_state2)
        machine.add_transition(mock_transition)

        # Initialization should fail due to invalid transition
        with self.assertRaises(ValueError) as cm:
            machine.initialize()

        self.assertIn("Invalid transition configuration", str(cm.exception))
        self.assertEqual(machine.status, MachineStatus.UNINITIALIZED)
        mock_transition.is_valid.assert_called_once()


class TestProtocolMachine(unittest.TestCase):
    """Test cases for the ProtocolMachine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.machine = ProtocolMachine("test_protocol")

        # Create mock states
        self.state1 = Mock(spec=State)
        self.state1.id = "state1"
        self.state1.initialize = Mock()
        self.state1.enter = Mock()
        self.state1.exit = Mock()
        self.state1.is_valid = Mock(return_value=True)

        self.state2 = Mock(spec=State)
        self.state2.id = "state2"
        self.state2.initialize = Mock()
        self.state2.enter = Mock()
        self.state2.exit = Mock()
        self.state2.is_valid = Mock(return_value=True)

        self.initial_state = Mock(spec=State)
        self.initial_state.id = "initial"
        self.initial_state.initialize = Mock()
        self.initial_state.enter = Mock()
        self.initial_state.exit = Mock()
        self.initial_state.is_valid = Mock(return_value=True)

        self.running_state = Mock(spec=State)
        self.running_state.id = "running"
        self.running_state.initialize = Mock()
        self.running_state.enter = Mock()
        self.running_state.exit = Mock()
        self.running_state.is_valid = Mock(return_value=True)

        self.stopped_state = Mock(spec=State)
        self.stopped_state.id = "stopped"
        self.stopped_state.initialize = Mock()
        self.stopped_state.enter = Mock()
        self.stopped_state.exit = Mock()
        self.stopped_state.is_valid = Mock(return_value=True)

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
        mock_state = Mock(spec=State)
        mock_state.id = "cyclic_state"
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
        mock_region = Mock(spec=Region)
        mock_region.id = "test_region"
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=True)
        self.machine.add_region(mock_region)

        with self.assertRaises(ValueError) as cm:
            self.machine.initialize()
        self.assertIn("cyclic submachine reference detected", str(cm.exception).lower())


class TestMachineBuilder(unittest.TestCase):
    """Test cases for the MachineBuilder class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.builder = MachineBuilder()

    def test_machine_configuration(self):
        """Test machine configuration and building."""
        # Set machine type
        self.builder.set_machine_type(BasicStateMachine)

        # Add components
        mock_state = Mock(spec=State)
        mock_region = Mock(spec=Region)
        self.builder.add_component("states", mock_state)
        self.builder.add_component("regions", mock_region)

        # Build machine
        machine = self.builder.build()
        self.assertIsInstance(machine, BasicStateMachine)
        # Verify components were added

    def test_dependency_tracking(self):
        """Test component dependency tracking."""
        self.builder.add_dependency("component1", "component2")
        self.assertIn("component2", self.builder.dependencies["component1"])

        # Test cyclic dependency detection
        with self.assertRaises(ValueError):
            self.builder.add_dependency("component2", "component1")

    def test_validation(self):
        """Test machine configuration validation."""
        # Test invalid machine type
        with self.assertRaises(ValueError):
            self.builder.set_machine_type(str)  # Not a StateMachine subclass

        # Test missing required components
        self.builder.set_machine_type(BasicStateMachine)
        with self.assertRaises(ValueError):
            self.builder.build()  # No components added

    def test_component_validation(self):
        """Test component validation during building."""
        # Set up mock components
        mock_state = Mock(spec=State)
        mock_state.id = "test_state"
        mock_state.initialize = Mock()
        mock_state.enter = Mock()
        mock_state.exit = Mock()
        mock_state.is_valid = Mock(return_value=False)

        mock_region = Mock(spec=Region)
        mock_region.id = "test_region"
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=True)

        mock_transition = Mock(spec=Transition)
        mock_transition.initialize = Mock()
        mock_transition.is_valid = Mock(return_value=True)

        # Test state validation
        self.builder.set_machine_type(BasicStateMachine)
        self.builder.add_component("states", mock_state)
        self.builder.add_component("regions", mock_region)  # Add required region
        with self.assertRaises(ValueError) as cm:
            self.builder.build()
        self.assertIn("Invalid state configuration", str(cm.exception))

        # Test region validation
        self.builder = MachineBuilder()  # Reset builder
        self.builder.set_machine_type(BasicStateMachine)
        mock_state.is_valid = Mock(return_value=True)  # Make state valid
        mock_region.is_valid = Mock(return_value=False)  # Make region invalid
        self.builder.add_component("states", mock_state)
        self.builder.add_component("regions", mock_region)
        with self.assertRaises(ValueError) as cm:
            self.builder.build()
        self.assertIn("Invalid region configuration", str(cm.exception))

        # Test transition validation
        self.builder = MachineBuilder()  # Reset builder
        self.builder.set_machine_type(BasicStateMachine)
        mock_region.is_valid = Mock(return_value=True)  # Make region valid
        mock_transition.is_valid = Mock(return_value=False)  # Make transition invalid
        self.builder.add_component("states", mock_state)
        self.builder.add_component("regions", mock_region)
        self.builder.add_component("transitions", mock_transition)
        with self.assertRaises(ValueError) as cm:
            self.builder.build()
        self.assertIn("Invalid transition configuration", str(cm.exception))

    def test_dependency_cycle_detection(self):
        """Test detection of dependency cycles."""
        builder = MachineBuilder()
        builder.set_machine_type(BasicStateMachine)
        
        # Create mock components
        mock_state1 = Mock(spec=State)
        mock_state1.id = "state1"
        mock_state1.initialize = Mock()
        mock_state1.enter = Mock()
        mock_state1.exit = Mock()
        mock_state1.is_valid = Mock(return_value=True)
        
        mock_state2 = Mock(spec=State)
        mock_state2.id = "state2"
        mock_state2.initialize = Mock()
        mock_state2.enter = Mock()
        mock_state2.exit = Mock()
        mock_state2.is_valid = Mock(return_value=True)
        
        mock_state3 = Mock(spec=State)
        mock_state3.id = "state3"
        mock_state3.initialize = Mock()
        mock_state3.enter = Mock()
        mock_state3.exit = Mock()
        mock_state3.is_valid = Mock(return_value=True)

        # Create required region
        mock_region = Mock(spec=Region)
        mock_region.id = "region1"
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=True)
        
        # Add components
        builder.add_component("states", mock_state1)
        builder.add_component("states", mock_state2)
        builder.add_component("states", mock_state3)
        builder.add_component("regions", mock_region)  # Add required region
        
        # Add dependencies that would create a cycle
        builder.add_dependency("state1", "state2")
        builder.add_dependency("state2", "state3")
        
        # This should fail as it creates a cycle
        with self.assertRaises(ValueError):
            builder.add_dependency("state3", "state1")
            
        # Build should still work without the cycle
        machine = builder.build()
        self.assertIsInstance(machine, BasicStateMachine)

    def test_unresolved_dependency_handling(self):
        """Test handling of unresolved dependencies."""
        builder = MachineBuilder()
        builder.set_machine_type(BasicStateMachine)
        
        # Create mock components
        mock_state1 = Mock(spec=State)
        mock_state1.id = "state1"
        mock_state1.initialize = Mock()
        mock_state1.enter = Mock()
        mock_state1.exit = Mock()
        mock_state1.is_valid = Mock(return_value=True)
        
        mock_state2 = Mock(spec=State)
        mock_state2.id = "state2"
        mock_state2.initialize = Mock()
        mock_state2.enter = Mock()
        mock_state2.exit = Mock()
        mock_state2.is_valid = Mock(return_value=True)

        # Create required region
        mock_region = Mock(spec=Region)
        mock_region.id = "region1"
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=True)
        
        # Add components
        builder.add_component("states", mock_state1)
        builder.add_component("states", mock_state2)
        builder.add_component("regions", mock_region)  # Add required region
        
        # Add dependency on non-existent component
        builder.add_dependency("state1", "non_existent")
        
        # Build should fail due to unresolved dependency
        with self.assertRaises(ValueError) as cm:
            builder.build()
            
        self.assertIn("Unresolved dependency", str(cm.exception))

    def test_component_validation_order(self):
        """Test component validation order and error handling."""
        builder = MachineBuilder()
        builder.set_machine_type(BasicStateMachine)
        
        # Create mock components
        mock_state = Mock(spec=State)
        mock_state.id = "state1"
        mock_state.initialize = Mock()
        mock_state.enter = Mock()
        mock_state.exit = Mock()
        mock_state.is_valid = Mock(return_value=True)
        
        mock_region = Mock(spec=Region)
        mock_region.id = "region1"
        mock_region.initialize = Mock()
        mock_region.activate = Mock()
        mock_region.deactivate = Mock()
        mock_region.is_valid = Mock(return_value=True)
        
        mock_transition = Mock(spec=Transition)
        mock_transition.source = mock_state
        mock_transition.target = mock_state
        mock_transition.initialize = Mock()
        mock_transition.is_valid = Mock(return_value=True)
        
        # Add components in wrong order
        builder.add_component("transitions", mock_transition)
        builder.add_component("regions", mock_region)
        builder.add_component("states", mock_state)
        
        # Build should still work as components are added in correct order internally
        machine = builder.build()
        self.assertIsInstance(machine, BasicStateMachine)
        
        # Test missing components
        builder = MachineBuilder()
        builder.set_machine_type(BasicStateMachine)
        
        # Add only transitions (no states or regions)
        builder.add_component("transitions", mock_transition)
        
        with self.assertRaises(ValueError) as cm:
            builder.build()
            
        self.assertIn("Missing required component types", str(cm.exception))

    def test_component_type_validation(self):
        """Test validation of component types."""
        builder = MachineBuilder()
        builder.set_machine_type(BasicStateMachine)
        
        # Create mock components
        mock_state = Mock()
        mock_state.id = "state1"
        mock_state.initialize = Mock()
        mock_state.is_valid = Mock(return_value=True)
        
        # Add component with invalid type
        builder.add_component("invalid_type", mock_state)
        
        # Build should still work as invalid type is ignored
        with self.assertRaises(ValueError):
            builder.build()  # Fails because no valid components added


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
