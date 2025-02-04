import unittest
from threading import Lock
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

from gotstate.core.event import Event, EventKind, EventPriority
from gotstate.core.machine.machine_status import MachineStatus
from gotstate.core.machine.state_machine import StateMachine
from gotstate.core.transition import Transition
from tests.unit.machine.machine_mocks import MockRegion, MockState


class TestStateMachineAbstract(unittest.TestCase):
    """Test that StateMachine properly raises NotImplementedError for abstract methods."""

    def setUp(self):
        """Set up test cases."""

        class TestMachine(StateMachine):
            def __init__(self):
                self._status = MachineStatus.UNINITIALIZED
                self._status_lock = Lock()
                self._collection_lock = Lock()
                self._states = {}
                self._regions = {}
                self._transitions = set()
                self._resources = set()

        self.machine = TestMachine()

    def test_initialize_raises_not_implemented(self):
        """Test that initialize raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.initialize()

    def test_activate_raises_not_implemented(self):
        """Test that activate raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.activate()

    def test_terminate_raises_not_implemented(self):
        """Test that terminate raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.terminate()

    def test_add_state_raises_not_implemented(self):
        """Test that add_state raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.add_state(Mock())

    def test_add_region_raises_not_implemented(self):
        """Test that add_region raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.add_region(Mock())

    def test_add_transition_raises_not_implemented(self):
        """Test that add_transition raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.add_transition(Mock())

    def test_add_resource_raises_not_implemented(self):
        """Test that add_resource raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.add_resource(Mock())

    def test_process_event_raises_not_implemented(self):
        """Test that process_event raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "StateMachine is an abstract base class"):
            self.machine.process_event(Mock())

    def test_initialize_components_raises_not_implemented(self):
        """Test that _initialize_components raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._initialize_components()

    def test_validate_configuration_raises_not_implemented(self):
        """Test that _validate_configuration raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._validate_configuration()

    def test_stop_components_raises_not_implemented(self):
        """Test that _stop_components raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._stop_components()

    def test_start_components_raises_not_implemented(self):
        """Test that _start_components raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._start_components()

    def test_start_event_processing_raises_not_implemented(self):
        """Test that _start_event_processing raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._start_event_processing()

    def test_stop_event_processing_raises_not_implemented(self):
        """Test that _stop_event_processing raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._stop_event_processing()

    def test_cleanup_resources_raises_not_implemented(self):
        """Test that _cleanup_resources raises NotImplementedError."""
        with self.assertRaisesRegex(NotImplementedError, "Must be implemented by subclasses"):
            self.machine._cleanup_resources()


class ConcreteStateMachine(StateMachine):
    """Concrete implementation of StateMachine for testing."""

    def __init__(self):
        """Initialize the concrete state machine."""
        self._status = MachineStatus.UNINITIALIZED
        self._status_lock = Lock()
        self._collection_lock = Lock()
        self._states = {}
        self._regions = {}
        self._transitions = set()
        self._resources = set()

    def initialize(self):
        """Initialize the machine."""
        with self._status_lock:
            if self._status != MachineStatus.UNINITIALIZED:
                raise RuntimeError("Machine must be in UNINITIALIZED status")
            self._status = MachineStatus.INITIALIZING
            try:
                self._validate_configuration()
                self._initialize_components()
                self._status = MachineStatus.ACTIVE
            except Exception as e:
                self._status = MachineStatus.UNINITIALIZED
                raise

    def activate(self):
        """Activate the machine."""
        with self._status_lock:
            if self._status != MachineStatus.ACTIVE:
                raise RuntimeError("Machine must be in ACTIVE status")
            self._start_components()
            self._start_event_processing()

    def terminate(self):
        """Terminate the machine."""
        with self._status_lock:
            if self._status != MachineStatus.ACTIVE:
                raise RuntimeError("Machine must be in ACTIVE status")
            try:
                self._status = MachineStatus.TERMINATING
                self._stop_components()
                self._stop_event_processing()
                self._cleanup_resources()
                self._status = MachineStatus.TERMINATED
            except Exception as e:
                if "Termination failed:" not in str(e):
                    raise RuntimeError(f"Termination failed: {e}")
                raise

    def add_state(self, state):
        """Add a state to the machine."""
        with self._collection_lock:
            if state.id in self._states:
                raise ValueError(f"State {state.id} already exists")
            self._states[state.id] = state

    def add_region(self, region):
        """Add a region to the machine."""
        with self._collection_lock:
            if region.id in self._regions:
                raise ValueError(f"Region {region.id} already exists")
            self._regions[region.id] = region

    def add_transition(self, transition):
        """Add a transition to the machine."""
        with self._collection_lock:
            self._transitions.add(transition)

    def add_resource(self, resource):
        """Add a resource to the machine."""
        with self._collection_lock:
            self._resources.add(resource)

    def process_event(self, event):
        """Process an event."""
        if self._status != MachineStatus.ACTIVE:
            raise RuntimeError("Machine must be active to process events")
        # Implementation would go here

    def _initialize_components(self):
        """Initialize components."""
        pass

    def _validate_configuration(self):
        """Validate configuration."""
        for state in self._states.values():
            if not state.is_valid():
                raise ValueError(f"Invalid state: {state}")
        for region in self._regions.values():
            if not region.is_valid():
                raise ValueError(f"Invalid region: {region}")
        invalid_transitions = [t for t in self._transitions if not t.is_valid()]
        if invalid_transitions:
            raise ValueError(f"Invalid transitions: {invalid_transitions}")

    def _stop_components(self):
        """Stop components."""
        pass

    def _start_components(self):
        """Start components."""
        pass

    def _start_event_processing(self):
        """Start event processing."""
        pass

    def _stop_event_processing(self):
        """Stop event processing."""
        pass

    def _cleanup_resources(self):
        """Clean up resources."""
        for resource in self._resources:
            if hasattr(resource, "cleanup"):
                resource.cleanup()


class TestStateMachineImplementation(unittest.TestCase):
    """Test concrete implementation of StateMachine."""

    def setUp(self):
        """Set up test cases."""
        self.machine = ConcreteStateMachine()

    def test_track_event_and_handle_component_error(self):
        """Test that _handle_component_error uses _track_event correctly."""

        # Create a test machine that tracks events
        class TrackingMachine(ConcreteStateMachine):
            def __init__(self):
                super().__init__()
                self.tracked_events = []

            def _track_event(self, event_name: str, data: Dict[str, Any]) -> None:
                self.tracked_events.append((event_name, data))

        machine = TrackingMachine()
        error = ValueError("test error")
        machine._handle_component_error("test_type", "test_id", error)

        self.assertEqual(len(machine.tracked_events), 1)
        event_name, data = machine.tracked_events[0]
        self.assertEqual(event_name, "component_stop_failure")
        self.assertEqual(data["component_type"], "test_type")
        self.assertEqual(data["component_id"], "test_id")
        self.assertEqual(data["error"], str(error))

    def test_cleanup_components(self):
        """Test that _cleanup_components clears all collections."""
        # Add some test components
        state = MockState("test_state")
        region = MockRegion("test_region")
        transition = Mock()
        self.machine._states = {"test_state": state}
        self.machine._regions = {"test_region": region}
        self.machine._transitions = {transition}

        # Verify components are present
        self.assertEqual(len(self.machine._states), 1)
        self.assertEqual(len(self.machine._regions), 1)
        self.assertEqual(len(self.machine._transitions), 1)

        # Call cleanup
        self.machine._cleanup_components()

        # Verify all collections are cleared
        self.assertEqual(len(self.machine._states), 0)
        self.assertEqual(len(self.machine._regions), 0)
        self.assertEqual(len(self.machine._transitions), 0)

    def test_initialize_success(self):
        """Test successful initialization."""
        self.machine.initialize()
        self.assertEqual(self.machine._status, MachineStatus.ACTIVE)

    def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        self.machine.initialize()  # First initialize
        with self.assertRaises(RuntimeError):
            self.machine.initialize()  # Second initialize should fail

    def test_initialize_validation_failure(self):
        """Test initialization with validation failure."""
        state = MockState("test_state")
        state.is_valid.return_value = False
        self.machine._states = {"test_state": state}
        with self.assertRaises(ValueError):
            self.machine.initialize()
        self.assertEqual(self.machine._status, MachineStatus.UNINITIALIZED)

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        state = MockState("test_state")
        region = MockRegion("test_region")
        self.machine._states = {"test_state": state}
        self.machine._regions = {"test_region": region}
        self.machine._validate_configuration()  # Should not raise

    def test_validate_configuration_invalid_state(self):
        """Test configuration validation with invalid state."""
        state = MockState("test_state")
        state.is_valid.return_value = False
        self.machine._states = {"test_state": state}
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Invalid state", str(ctx.exception))

    def test_validate_configuration_invalid_region(self):
        """Test configuration validation with invalid region."""
        region = MockRegion("test_region")
        region.is_valid.return_value = False
        self.machine._regions = {"test_region": region}
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Invalid region", str(ctx.exception))

    def test_validate_configuration_invalid_transition(self):
        """Test configuration validation with invalid transition."""
        transition = Mock()
        transition.is_valid = Mock(return_value=False)
        self.machine._transitions = {transition}
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Invalid transitions", str(ctx.exception))

    def test_cleanup_resources_success(self):
        """Test successful resource cleanup."""
        resource = Mock()
        resource.cleanup = Mock()
        self.machine._resources.add(resource)
        self.machine._cleanup_resources()
        resource.cleanup.assert_called_once()

    def test_cleanup_resources_no_cleanup_method(self):
        """Test resource cleanup when resource has no cleanup method."""
        resource = Mock()
        del resource.cleanup  # Remove cleanup method
        self.machine._resources.add(resource)
        self.machine._cleanup_resources()  # Should not raise

    def test_cleanup_resources_failure(self):
        """Test resource cleanup with failure."""
        resource = Mock()
        resource.cleanup = Mock(side_effect=Exception("Cleanup failed"))
        self.machine._resources.add(resource)
        with self.assertRaises(Exception) as ctx:
            self.machine._cleanup_resources()
        self.assertEqual(str(ctx.exception), "Cleanup failed")

    def test_terminate_success(self):
        """Test successful termination."""
        self.machine._status = MachineStatus.ACTIVE
        self.machine.terminate()
        self.assertEqual(self.machine._status, MachineStatus.TERMINATED)

    def test_terminate_already_terminated(self):
        """Test termination when already terminated."""
        self.machine._status = MachineStatus.TERMINATED
        with self.assertRaises(RuntimeError):
            self.machine.terminate()

    def test_terminate_with_unwrapped_error(self):
        """Test termination with an unwrapped error that needs wrapping."""
        self.machine._status = MachineStatus.ACTIVE
        original_error = ValueError("Original error")

        with patch.object(self.machine, "_stop_components") as mock_stop:
            mock_stop.side_effect = original_error
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            self.assertEqual(str(ctx.exception), f"Termination failed: {original_error}")

    def test_terminate_with_wrapped_error(self):
        """Test termination with already wrapped error."""
        self.machine._status = MachineStatus.ACTIVE
        wrapped_error = RuntimeError("Termination failed: inner error")

        with patch.object(self.machine, "_stop_components") as mock_stop:
            mock_stop.side_effect = wrapped_error
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            self.assertEqual(str(ctx.exception), str(wrapped_error))

    def test_terminate_with_mixed_error_types(self):
        """Test termination with different error types."""
        self.machine._status = MachineStatus.ACTIVE
        value_error = ValueError("Value error")

        with patch.object(self.machine, "_stop_components") as mock_stop:
            mock_stop.side_effect = value_error
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            self.assertEqual(str(ctx.exception), f"Termination failed: {value_error}")

    def test_terminate_with_component_and_cleanup_errors(self):
        """Test termination with both component and cleanup errors."""
        self.machine._status = MachineStatus.ACTIVE
        component_error = RuntimeError("Component stop failed")
        cleanup_error = RuntimeError("Cleanup failed")

        with patch.object(self.machine, "_stop_components") as mock_stop, patch.object(
            self.machine, "_cleanup_resources"
        ) as mock_cleanup:
            mock_stop.side_effect = component_error
            mock_cleanup.side_effect = cleanup_error
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            self.assertEqual(str(ctx.exception), f"Termination failed: {component_error}")

    def test_terminate_with_double_wrapped_error(self):
        """Test termination with double wrapped error."""
        self.machine._status = MachineStatus.ACTIVE
        inner_error = RuntimeError("Inner error")
        wrapped_error = RuntimeError(f"Termination failed: {inner_error}")

        with patch.object(self.machine, "_stop_components") as mock_stop:
            mock_stop.side_effect = wrapped_error
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            self.assertEqual(str(ctx.exception), str(wrapped_error))


if __name__ == "__main__":
    unittest.main()
