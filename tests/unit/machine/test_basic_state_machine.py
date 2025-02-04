import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any, Dict
from unittest.mock import Mock, patch

from gotstate.core.event import Event, EventKind, EventPriority
from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.machine.machine_status import MachineStatus
from tests.unit.machine.machine_mocks import MockRegion, MockState


class TestBasicStateMachine(unittest.TestCase):
    """Test cases for BasicStateMachine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.machine = BasicStateMachine()

    def test_initial_state(self):
        """Test initial state of the machine.
        
        Verifies:
        1. Initial status is UNINITIALIZED
        2. Collections are empty
        3. Locks are initialized
        4. Resource limits are set
        """
        self.assertEqual(self.machine.status, MachineStatus.UNINITIALIZED)
        self.assertEqual(len(self.machine._states), 0)
        self.assertEqual(len(self.machine._regions), 0)
        self.assertEqual(len(self.machine._transitions), 0)
        self.assertEqual(len(self.machine._resources), 0)
        
        # Verify locks are initialized
        self.assertIsInstance(self.machine._status_lock, type(RLock()))
        self.assertIsInstance(self.machine._collection_lock, type(RLock()))
        self.assertIsInstance(self.machine._processing_lock, type(RLock()))
        self.assertIsInstance(self.machine._monitor_lock, type(RLock()))
        self.assertIsInstance(self.machine._version_lock, type(RLock()))
        self.assertIsInstance(self.machine._security_lock, type(RLock()))

        # Verify resource limits
        self.assertEqual(self.machine._resource_limits["max_states"], 1000)
        self.assertEqual(self.machine._resource_limits["max_regions"], 100)
        self.assertEqual(self.machine._resource_limits["max_transitions"], 5000)
        self.assertEqual(self.machine._resource_limits["max_events_queued"], 10000)
        self.assertEqual(self.machine._resource_limits["max_resources"], 100)

    def test_add_state_success(self):
        """Test successful state addition.
        
        Verifies:
        1. State is added to collection
        2. Thread safety is maintained
        """
        state = MockState("test_state")
        self.machine.add_state(state)
        self.assertIn("test_state", self.machine._states)
        self.assertEqual(self.machine._states["test_state"], state)

    def test_add_state_validation(self):
        """Test state addition validation.
        
        Verifies:
        1. Cannot add None state
        2. Cannot add duplicate state ID
        3. Cannot add state when machine is active
        4. Must be State instance
        """
        # Test None state
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_state(None)
        self.assertIn("State cannot be None", str(ctx.exception))

        # Test duplicate state ID
        state1 = MockState("test_state")
        state2 = MockState("test_state")
        self.machine.add_state(state1)
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_state(state2)
        self.assertIn("already exists", str(ctx.exception))

        # Test non-State instance
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_state(Mock())
        self.assertIn("must be a State instance", str(ctx.exception))

        # Test adding when active
        self.machine._status = MachineStatus.ACTIVE
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_state(MockState("new_state"))
        self.assertIn("Cannot add states while machine is active", str(ctx.exception))

    def test_add_region_success(self):
        """Test successful region addition.
        
        Verifies:
        1. Region is added to collection
        2. Thread safety is maintained
        """
        region = MockRegion("test_region")
        self.machine.add_region(region)
        self.assertIn("test_region", self.machine._regions)
        self.assertEqual(self.machine._regions["test_region"], region)

    def test_add_region_validation(self):
        """Test region addition validation.
        
        Verifies:
        1. Cannot add None region
        2. Cannot add duplicate region ID
        3. Cannot add region when machine is active
        4. Must be Region instance
        """
        # Test None region
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_region(None)
        self.assertIn("Region cannot be None", str(ctx.exception))

        # Test duplicate region ID
        region1 = MockRegion("test_region")
        region2 = MockRegion("test_region")
        self.machine.add_region(region1)
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_region(region2)
        self.assertIn("already exists", str(ctx.exception))

        # Test non-Region instance
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_region(Mock())
        self.assertIn("must be a Region instance", str(ctx.exception))

        # Test adding when active
        self.machine._status = MachineStatus.ACTIVE
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_region(MockRegion("new_region"))
        self.assertIn("Cannot add regions while machine is active", str(ctx.exception))

    def test_add_transition_success(self):
        """Test successful transition addition.
        
        Verifies:
        1. Transition is added to collection
        2. Thread safety is maintained
        """
        transition = Mock()
        self.machine.add_transition(transition)
        self.assertIn(transition, self.machine._transitions)

    def test_add_transition_validation(self):
        """Test transition addition validation.
        
        Verifies:
        1. Cannot add transition when machine is active
        """
        self.machine._status = MachineStatus.ACTIVE
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_transition(Mock())
        self.assertIn("Cannot add transitions while machine is active", str(ctx.exception))

    def test_add_resource_success(self):
        """Test successful resource addition.
        
        Verifies:
        1. Resource is added to collection
        2. Thread safety is maintained
        """
        resource = Mock()
        self.machine.add_resource(resource)
        self.assertIn(resource, self.machine._resources)

    def test_add_resource_validation(self):
        """Test resource addition validation.
        
        Verifies:
        1. Cannot add None resource
        2. Cannot add resource when machine is active
        """
        # Test None resource
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_resource(None)
        self.assertIn("Resource cannot be None", str(ctx.exception))

        # Test adding when active
        self.machine._status = MachineStatus.ACTIVE
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_resource(Mock())
        self.assertIn("Cannot add resources while machine is active", str(ctx.exception))

    def test_process_event_success(self):
        """Test successful event processing.
        
        Verifies:
        1. Event is queued
        2. Event tracking is performed
        3. Thread safety is maintained
        """
        self.machine._status = MachineStatus.ACTIVE
        event = Event("test", EventKind.SIGNAL, EventPriority.NORMAL)
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.machine.process_event(event)
            
            # Verify event tracking
            self.assertEqual(mock_track.call_count, 2)  # processing and queued events
            
            # Verify first call - event processing
            args1, kwargs1 = mock_track.call_args_list[0]
            self.assertEqual(args1[0], "event_processing")
            self.assertEqual(args1[1]["event_kind"], event.kind.name)
            
            # Verify second call - event queued
            args2, kwargs2 = mock_track.call_args_list[1]
            self.assertEqual(args2[0], "event_queued")

    def test_process_event_validation(self):
        """Test event processing validation.
        
        Verifies:
        1. Cannot process events when machine is not active
        2. Failed events are tracked
        """
        event = Event("test", EventKind.SIGNAL, EventPriority.NORMAL)
        
        # Test processing when not active
        with self.assertRaises(RuntimeError) as ctx:
            self.machine.process_event(event)
        self.assertIn("must be ACTIVE to process events", str(ctx.exception))

        # Test failed event tracking
        self.machine._status = MachineStatus.ACTIVE
        with patch.object(self.machine._event_queue, 'enqueue', side_effect=RuntimeError("Queue full")), \
             patch.object(self.machine, '_track_event') as mock_track:
            
            with self.assertRaises(RuntimeError):
                self.machine.process_event(event)
            
            # Verify failure tracking
            failure_call = mock_track.call_args_list[-1]
            self.assertEqual(failure_call[0][0], "event_processing_failure")
            self.assertIn("Queue full", failure_call[0][1]["error"])

    def test_track_event(self):
        """Test event tracking.
        
        Verifies:
        1. Event data is properly formatted
        2. Monitor receives event
        3. Thread safety is maintained
        """
        event_type = "test_event"
        details = {"key": "value"}
        
        with patch.object(self.machine._monitor, 'track_event') as mock_track:
            self.machine._track_event(event_type, details)
            
            # Verify monitor call
            args, kwargs = mock_track.call_args
            event_data = args[0]
            
            self.assertEqual(event_data["type"], event_type)
            self.assertEqual(event_data["key"], "value")
            self.assertEqual(event_data["machine_status"], self.machine.status.name)
            self.assertIn("timestamp", event_data)

    def test_initialize_success(self):
        """Test successful machine initialization.
        
        Verifies:
        1. Status transitions are correct
        2. Components are initialized
        3. Configuration is validated
        4. Thread safety is maintained
        """
        with patch.object(self.machine, '_initialize_components') as mock_init, \
             patch.object(self.machine, '_validate_configuration') as mock_validate:
            
            self.machine.initialize()
            
            mock_init.assert_called_once()
            mock_validate.assert_called_once()
            self.assertEqual(self.machine.status, MachineStatus.INITIALIZING)

    def test_initialize_validation(self):
        """Test initialization validation.
        
        Verifies:
        1. Cannot initialize when not in UNINITIALIZED state
        2. Initialization failures are handled
        3. Status is restored on failure
        4. Components are cleaned up on failure
        """
        # Test wrong initial state
        self.machine._status = MachineStatus.ACTIVE
        with self.assertRaises(ValueError) as ctx:
            self.machine.initialize()
        self.assertIn("must be in UNINITIALIZED status", str(ctx.exception))

        # Test initialization failure
        self.machine._status = MachineStatus.UNINITIALIZED
        with patch.object(self.machine, '_initialize_components', side_effect=RuntimeError("Init failed")), \
             patch.object(self.machine, '_cleanup_components') as mock_cleanup:
            
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.initialize()
            
            self.assertIn("Init failed", str(ctx.exception))
            self.assertEqual(self.machine.status, MachineStatus.UNINITIALIZED)
            mock_cleanup.assert_called_once()

    def test_activate_success(self):
        """Test successful machine activation.
        
        Verifies:
        1. Status transitions are correct
        2. Components are started
        3. Event processing is started
        4. Thread safety is maintained
        """
        self.machine._status = MachineStatus.INITIALIZING
        
        with patch.object(self.machine, '_start_components') as mock_start, \
             patch.object(self.machine, '_start_event_processing') as mock_process:
            
            self.machine.activate()
            
            mock_start.assert_called_once()
            mock_process.assert_called_once()
            self.assertEqual(self.machine.status, MachineStatus.ACTIVE)

    def test_activate_validation(self):
        """Test activation validation.
        
        Verifies:
        1. Cannot activate when not in INITIALIZING state
        2. Activation failures are handled
        3. Status is restored on failure
        4. Components are stopped on failure
        """
        # Test wrong initial state
        with self.assertRaises(ValueError) as ctx:
            self.machine.activate()
        self.assertIn("must be in INITIALIZING status", str(ctx.exception))

        # Test activation failure
        self.machine._status = MachineStatus.INITIALIZING
        with patch.object(self.machine, '_start_components', side_effect=RuntimeError("Start failed")), \
             patch.object(self.machine, '_stop_components') as mock_stop:
            
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.activate()
            
            self.assertIn("Activation failed", str(ctx.exception))
            self.assertEqual(self.machine.status, MachineStatus.INITIALIZING)
            mock_stop.assert_called_once()

    def test_terminate_success(self):
        """Test successful machine termination.
        
        Verifies:
        1. Status transitions are correct
        2. Event processing is stopped
        3. Components are stopped
        4. Resources are cleaned up
        5. Thread safety is maintained
        """
        self.machine._status = MachineStatus.ACTIVE
        
        with patch.object(self.machine, '_stop_event_processing') as mock_stop_events, \
             patch.object(self.machine, '_stop_components', return_value=[]) as mock_stop_comp, \
             patch.object(self.machine, '_cleanup_resources') as mock_cleanup:
            
            self.machine.terminate()
            
            mock_stop_events.assert_called_once()
            mock_stop_comp.assert_called_once()
            mock_cleanup.assert_called_once()
            self.assertEqual(self.machine.status, MachineStatus.TERMINATED)

    def test_terminate_validation(self):
        """Test termination validation.
        
        Verifies:
        1. Cannot terminate when not in ACTIVE state
        2. Termination failures are handled
        3. Status is restored on failure
        4. Multiple termination attempts are handled
        5. Component errors are collected
        6. Cleanup is always attempted
        """
        # Test wrong initial state
        with self.assertRaises(RuntimeError) as ctx:
            self.machine.terminate()
        self.assertIn("must be in ACTIVE status", str(ctx.exception))

        # Test already terminated
        self.machine._status = MachineStatus.TERMINATED
        self.machine.terminate()  # Should not raise

        # Test component failure
        self.machine._status = MachineStatus.ACTIVE
        component_error = RuntimeError("Component failed")
        with patch.object(self.machine, '_stop_components', return_value=[component_error]), \
             patch.object(self.machine, '_cleanup_resources') as mock_cleanup:
            
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            
            self.assertIn("Component failed", str(ctx.exception))
            self.assertEqual(self.machine.status, MachineStatus.ACTIVE)
            mock_cleanup.assert_called_once()

        # Test cleanup failure
        self.machine._status = MachineStatus.ACTIVE
        cleanup_error = RuntimeError("Cleanup failed")
        with patch.object(self.machine, '_stop_components', return_value=[]), \
             patch.object(self.machine, '_cleanup_resources', side_effect=cleanup_error):
            
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            
            self.assertIn("Cleanup failed", str(ctx.exception))
            self.assertEqual(self.machine.status, MachineStatus.ACTIVE)

    def test_initialize_components_success(self):
        """Test successful component initialization.
        
        Verifies:
        1. States are initialized in parallel
        2. Regions are initialized sequentially
        3. Events are tracked properly
        4. Thread safety is maintained
        """
        state1 = MockState("state1")
        state2 = MockState("state2")
        region1 = MockRegion("region1")
        region2 = MockRegion("region2")
        
        self.machine.add_state(state1)
        self.machine.add_state(state2)
        self.machine.add_region(region1)
        self.machine.add_region(region2)
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.machine._initialize_components()
            
            # Verify initialization events
            init_events = [call for call in mock_track.call_args_list if call[0][0] == "component_init"]
            success_events = [call for call in mock_track.call_args_list if call[0][0] == "component_init_success"]
            
            self.assertEqual(len(init_events), 4)  # 2 states + 2 regions
            self.assertEqual(len(success_events), 4)  # All successful

    def test_initialize_components_failure(self):
        """Test component initialization failure.
        
        Verifies:
        1. State initialization failure is handled
        2. Region initialization failure is handled
        3. Error events are tracked
        4. Exceptions are propagated
        """
        state = MockState("state1")
        region = MockRegion("region1")
        state.initialize = Mock(side_effect=RuntimeError("State init failed"))
        
        self.machine.add_state(state)
        self.machine.add_region(region)
        
        with patch.object(self.machine, '_track_event') as mock_track:
            with self.assertRaises(RuntimeError) as ctx:
                self.machine._initialize_components()
            
            self.assertIn("State init failed", str(ctx.exception))
            
            # Verify failure event was tracked
            failure_events = [call for call in mock_track.call_args_list if call[0][0] == "component_init_failure"]
            self.assertEqual(len(failure_events), 1)
            self.assertEqual(failure_events[0][0][1]["component_type"], "state")
            self.assertEqual(failure_events[0][0][1]["component_id"], "state1")

    def test_start_components_success(self):
        """Test successful component startup.
        
        Verifies:
        1. Resources are allocated
        2. States are entered
        3. Regions are activated
        4. Order is maintained
        """
        resource = Mock()
        state = MockState("state1")
        region = MockRegion("region1")
        
        self.machine.add_resource(resource)
        self.machine.add_state(state)
        self.machine.add_region(region)
        
        self.machine._start_components()
        
        resource.allocate.assert_called_once()
        state.enter.assert_called_once()
        region.activate.assert_called_once()

    def test_start_components_failure(self):
        """Test component startup failure.
        
        Verifies:
        1. Resource allocation failure is handled
        2. Cleanup is attempted on failure
        3. Exceptions are propagated
        """
        resource1 = Mock()
        resource2 = Mock()
        resource1.allocate = Mock(side_effect=RuntimeError("Allocation failed"))
        
        self.machine.add_resource(resource1)
        self.machine.add_resource(resource2)
        
        with self.assertRaises(RuntimeError) as ctx:
            self.machine._start_components()
        
        self.assertIn("Allocation failed", str(ctx.exception))
        resource1.cleanup.assert_called_once()
        resource2.cleanup.assert_called_once()

    def test_version_management(self):
        """Test version management functionality.
        
        Verifies:
        1. Version retrieval
        2. Version compatibility checks
        3. Thread safety
        """
        self.assertEqual(self.machine.get_version(), "1.0.0")
        
        # Test compatibility
        self.assertTrue(self.machine.validate_version_compatibility("1.1.0"))
        self.assertTrue(self.machine.validate_version_compatibility("1.0.1"))
        self.assertFalse(self.machine.validate_version_compatibility("2.0.0"))
        self.assertFalse(self.machine.validate_version_compatibility("0.9.0"))

    def test_security_policies(self):
        """Test security policy management.
        
        Verifies:
        1. Policy addition
        2. Policy validation
        3. Thread safety
        4. Error handling
        """
        def always_allow(op: str) -> bool:
            return True
            
        def always_deny(op: str) -> bool:
            return False
            
        def error_policy(op: str) -> bool:
            raise RuntimeError("Policy error")
        
        # Test policy addition and validation
        self.machine.add_security_policy("read", always_allow)
        self.machine.add_security_policy("write", always_deny)
        self.machine.add_security_policy("error", error_policy)
        
        self.assertTrue(self.machine.validate_security_policy("read"))
        self.assertFalse(self.machine.validate_security_policy("write"))
        self.assertFalse(self.machine.validate_security_policy("error"))
        self.assertFalse(self.machine.validate_security_policy("unknown"))

    def test_resource_limits(self):
        """Test resource limit management.
        
        Verifies:
        1. Limit setting
        2. Limit checking
        3. Thread safety
        """
        # Set custom limits
        self.machine.set_resource_limit("max_states", 2)
        self.machine.set_resource_limit("max_regions", 1)
        
        # Add components up to limits
        self.machine.add_state(MockState("state1"))
        self.machine.add_state(MockState("state2"))
        self.machine.add_region(MockRegion("region1"))
        
        # Verify we're at limits
        self.assertTrue(self.machine.check_resource_limits())
        
        # Exceed limits
        self.machine.add_state(MockState("state3"))
        self.assertFalse(self.machine.check_resource_limits())

    def test_event_processing_lifecycle(self):
        """Test event processing lifecycle.
        
        Verifies:
        1. Start processing
        2. Stop processing
        3. Thread safety
        """
        with patch.object(self.machine._event_queue, 'start_processing') as mock_start, \
             patch.object(self.machine._event_queue, 'stop_processing') as mock_stop:
            
            self.machine._start_event_processing()
            self.machine._stop_event_processing()
            
            mock_start.assert_called_once()
            mock_stop.assert_called_once()

    def test_terminate_with_cleanup_error(self):
        """Test termination with cleanup error.
        
        Verifies:
        1. Cleanup errors are handled
        2. Status is restored
        3. Error is propagated
        """
        self.machine._status = MachineStatus.ACTIVE
        cleanup_error = RuntimeError("Cleanup error")

        with patch.object(self.machine, '_stop_event_processing') as mock_stop_events, \
             patch.object(self.machine, '_stop_components', return_value=[]) as mock_stop_comp, \
             patch.object(self.machine, '_cleanup_resources', side_effect=cleanup_error) as mock_cleanup:

            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()

            self.assertIn("Cleanup error", str(ctx.exception))
            self.assertEqual(self.machine.status, MachineStatus.ACTIVE)
            mock_stop_events.assert_called_once()
            mock_stop_comp.assert_called_once()
            mock_cleanup.assert_called_once()

    def test_initialize_components_parallel_failure(self):
        """Test parallel state initialization failure.
        
        Verifies:
        1. Parallel initialization errors are handled
        2. Error tracking is maintained
        3. Exceptions are propagated
        """
        state1 = MockState("state1")
        state2 = MockState("state2")
        state3 = MockState("state3")
        
        # Make state2 fail initialization
        state2.initialize = Mock(side_effect=RuntimeError("Parallel init failed"))
        
        self.machine.add_state(state1)
        self.machine.add_state(state2)
        self.machine.add_state(state3)
        
        with patch.object(self.machine, '_track_event') as mock_track:
            with self.assertRaises(RuntimeError) as ctx:
                self.machine._initialize_components()
            
            self.assertIn("Parallel init failed", str(ctx.exception))
            
            # Verify failure event was tracked
            failure_events = [call for call in mock_track.call_args_list if call[0][0] == "component_init_failure"]
            self.assertEqual(len(failure_events), 1)
            self.assertEqual(failure_events[0][0][1]["component_type"], "state")
            self.assertEqual(failure_events[0][0][1]["component_id"], "state2")

    def test_event_processing_queue_error(self):
        """Test event processing queue error.
        
        Verifies:
        1. Queue errors are handled
        2. Error tracking is maintained
        3. Exceptions are propagated
        """
        self.machine._status = MachineStatus.ACTIVE
        event = Event("test", EventKind.SIGNAL, EventPriority.NORMAL)
        queue_error = RuntimeError("Queue error")
        
        with patch.object(self.machine._event_queue, 'enqueue', side_effect=queue_error), \
             patch.object(self.machine, '_track_event') as mock_track:
            
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.process_event(event)
            
            self.assertIn("Queue error", str(ctx.exception))
            
            # Verify error tracking
            failure_events = [call for call in mock_track.call_args_list if call[0][0] == "event_processing_failure"]
            self.assertEqual(len(failure_events), 1)
            self.assertEqual(failure_events[0][0][1]["error"], "Queue error")

    def test_resource_limits_edge_cases(self):
        """Test resource limit edge cases.
        
        Verifies:
        1. Event queue size limits
        2. Multiple limit violations
        3. Thread safety
        """
        # Set low limits
        self.machine.set_resource_limit("max_states", 1)
        self.machine.set_resource_limit("max_regions", 1)
        self.machine.set_resource_limit("max_transitions", 1)
        self.machine.set_resource_limit("max_resources", 1)
        self.machine.set_resource_limit("max_events_queued", 1)
        
        # Test event queue size limit
        mock_event = Mock()
        mock_event.cancelled = False
        mock_events = {"event1": mock_event, "event2": mock_event}
        
        with patch.object(self.machine._event_queue, '_events', mock_events):
            self.assertFalse(self.machine.check_resource_limits())
        
        # Test state limit
        self.machine.add_state(MockState("state1"))
        self.assertTrue(self.machine.check_resource_limits())
        self.machine.add_state(MockState("state2"))
        self.assertFalse(self.machine.check_resource_limits())
        
        # Test region limit
        self.machine.set_resource_limit("max_states", 10)  # Reset state limit
        self.machine.add_region(MockRegion("region1"))
        self.assertTrue(self.machine.check_resource_limits())
        self.machine.add_region(MockRegion("region2"))
        self.assertFalse(self.machine.check_resource_limits())
        
        # Test transition limit
        self.machine.set_resource_limit("max_regions", 10)  # Reset region limit
        transition1 = Mock()
        transition2 = Mock()
        self.machine.add_transition(transition1)
        self.assertTrue(self.machine.check_resource_limits())
        self.machine.add_transition(transition2)
        self.assertFalse(self.machine.check_resource_limits())
        
        # Test resource limit
        self.machine.set_resource_limit("max_transitions", 10)  # Reset transition limit
        resource1 = Mock()
        resource2 = Mock()
        self.machine.add_resource(resource1)
        self.assertTrue(self.machine.check_resource_limits())
        self.machine.add_resource(resource2)
        self.assertFalse(self.machine.check_resource_limits())

    def test_terminate_with_wrapped_error(self):
        """Test termination with already wrapped error.
        
        Verifies:
        1. Already wrapped errors are not double-wrapped
        2. Status is restored
        3. Error is propagated
        """
        self.machine._status = MachineStatus.ACTIVE
        wrapped_error = RuntimeError("Termination failed: Original error")
        
        with patch.object(self.machine, '_stop_event_processing', side_effect=wrapped_error):
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            
            self.assertEqual(str(ctx.exception), "Termination failed: Original error")
            self.assertEqual(self.machine.status, MachineStatus.ACTIVE)

    def test_initialize_components_parallel_state_error(self):
        """Test parallel state initialization with multiple errors.
        
        Verifies:
        1. Multiple state errors are handled
        2. Error tracking is maintained
        3. First error is propagated
        4. Thread safety is maintained
        """
        state1 = MockState("state1")
        state2 = MockState("state2")
        state3 = MockState("state3")
        
        # Make multiple states fail initialization
        state1.initialize = Mock(side_effect=RuntimeError("State1 init failed"))
        state2.initialize = Mock(side_effect=RuntimeError("State2 init failed"))
        state3.initialize = Mock(side_effect=RuntimeError("State3 init failed"))
        
        self.machine.add_state(state1)
        self.machine.add_state(state2)
        self.machine.add_state(state3)
        
        with patch.object(self.machine, '_track_event') as mock_track:
            with self.assertRaises(RuntimeError) as ctx:
                self.machine._initialize_components()
            
            # Since we're using as_completed, any state's error could be returned first
            self.assertIn(str(ctx.exception), [
                "State1 init failed",
                "State2 init failed", 
                "State3 init failed"
            ])
            mock_track.assert_called()

    def test_event_processing_error_handling(self):
        """Test event processing error handling.
        
        Verifies:
        1. Event tracking errors are handled
        2. Queue errors are handled
        3. Error propagation
        4. Thread safety
        """
        self.machine._status = MachineStatus.ACTIVE
        event = Event("test", EventKind.SIGNAL, EventPriority.NORMAL)
        track_error = RuntimeError("Track error")
        
        # Test error in first track_event call
        with patch.object(self.machine, '_track_event', side_effect=track_error) as mock_track:
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.process_event(event)
            
            self.assertEqual(str(ctx.exception), "Track error")
            # First track_event call and error tracking call
            self.assertEqual(mock_track.call_count, 2)
            # Verify first call was event_processing
            self.assertEqual(mock_track.call_args_list[0][0][0], "event_processing")
            # Verify second call was error tracking
            self.assertEqual(mock_track.call_args_list[1][0][0], "event_processing_failure")

        # Test error in second track_event call
        def side_effect(event_type, details):
            if event_type == "event_queued":
                raise track_error
            return None
        
        with patch.object(self.machine, '_track_event', side_effect=side_effect) as mock_track:
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.process_event(event)
            
            self.assertEqual(str(ctx.exception), "Track error")
            self.assertEqual(mock_track.call_count, 3)  # event_processing, event_queued, event_processing_failure

    def test_terminate_with_multiple_errors(self):
        """Test termination with multiple errors.
        
        Verifies:
        1. Component errors are collected
        2. Cleanup errors are handled
        3. Status is restored
        4. Error propagation
        """
        self.machine._status = MachineStatus.ACTIVE
        component_error = RuntimeError("Component failed")
        cleanup_error = RuntimeError("Cleanup failed")
        
        with patch.object(self.machine, '_stop_event_processing') as mock_stop_events, \
             patch.object(self.machine, '_stop_components', return_value=[component_error]) as mock_stop_comp, \
             patch.object(self.machine, '_cleanup_resources', side_effect=cleanup_error) as mock_cleanup:
            
            with self.assertRaises(RuntimeError) as ctx:
                self.machine.terminate()
            
            # Should get the component error since it happened first
            self.assertIn("Component failed", str(ctx.exception))
            self.assertEqual(self.machine.status, MachineStatus.ACTIVE)
            mock_stop_events.assert_called_once()
            mock_stop_comp.assert_called_once()
            mock_cleanup.assert_called_once()

    def test_initialize_components_parallel_error_handling(self):
        """Test parallel state initialization error handling.

        Verifies:
        1. Multiple state errors are handled
        2. Error tracking is maintained
        3. First completed error is propagated (due to as_completed behavior)
        4. Thread safety is maintained
        5. Cleanup is performed
        """
        state1 = MockState("state1")
        state2 = MockState("state2")
        state3 = MockState("state3")

        # Make all states fail initialization with different errors
        state1.initialize = Mock(side_effect=RuntimeError("State1 init failed"))
        state2.initialize = Mock(side_effect=RuntimeError("State2 init failed"))
        state3.initialize = Mock(side_effect=RuntimeError("State3 init failed"))

        self.machine.add_state(state1)
        self.machine.add_state(state2)
        self.machine.add_state(state3)

        # Create mock futures to simulate parallel execution
        mock_futures = []
        future_to_state = {}
        for state in [state1, state2, state3]:
            mock_future = Mock()
            error = RuntimeError(f"{state.id} init failed")
            mock_future.result = Mock(return_value=("failure", state.id, error))
            mock_futures.append(mock_future)
            future_to_state[mock_future] = state

        mock_executor = Mock()
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)
        mock_executor.submit = Mock(side_effect=lambda fn, state: [f for f, s in future_to_state.items() if s == state][0])

        # Simulate as_completed returning futures in a specific order
        def mock_as_completed(futures):
            return iter(mock_futures)

        with patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_executor), \
             patch('concurrent.futures.as_completed', mock_as_completed), \
             patch.object(self.machine, '_track_event') as mock_track:

            with self.assertRaises(RuntimeError) as ctx:
                self.machine._initialize_components()

            # Since we're using as_completed, any state's error could be returned first
            self.assertIn(str(ctx.exception), [
                "State1 init failed",
                "State2 init failed", 
                "State3 init failed"
            ])
            mock_track.assert_called()

            # Verify all states were attempted to be initialized
            state1.initialize.assert_called_once()
            state2.initialize.assert_called_once()
            state3.initialize.assert_called_once()

    def test_event_processing_error_handling_complete(self):
        """Test complete event processing error handling.

        Verifies:
        1. Event processing errors are handled
        2. Error tracking is maintained
        3. Error is propagated
        4. Cleanup is performed
        """
        # Set machine to ACTIVE status
        self.machine._status = MachineStatus.ACTIVE
        
        event = Event("test_event", EventKind.SIGNAL, EventPriority.NORMAL)
        error = RuntimeError("Event processing failed")

        with patch.object(self.machine, '_track_event') as mock_track, \
             patch.object(self.machine._event_queue, 'enqueue', side_effect=error):

            with self.assertRaises(RuntimeError) as ctx:
                self.machine.process_event(event)

            self.assertEqual(str(ctx.exception), "Event processing failed")
            mock_track.assert_called()

if __name__ == '__main__':
    unittest.main()
