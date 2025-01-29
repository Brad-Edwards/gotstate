"""Unit tests for the StateMachine class.

Tests the core functionality of the StateMachine class including:
- Initialization and configuration
- Lifecycle management
- Component coordination
- Dynamic modifications
- Resource management
"""

import pytest
from unittest.mock import Mock, patch

from gotstate.core.machine import (
    StateMachine,
    BasicStateMachine,
    MachineStatus,
    MachineBuilder,
    MachineModifier,
    MachineMonitor,
    ProtocolMachine,
    SubmachineMachine
)
from gotstate.core.state import State, CompositeState
from gotstate.core.event import Event, EventQueue, EventKind, EventPriority
from gotstate.core.transition import Transition
from gotstate.core.region import Region
from gotstate.core.types import StateType, TransitionKind, TransitionPriority


def test_abstract_machine_initialization():
    """Test that abstract base class raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        StateMachine()


def test_basic_machine_initialization():
    """Test basic machine initialization."""
    machine = BasicStateMachine()
    assert machine.status == MachineStatus.UNINITIALIZED


def test_basic_machine_lifecycle():
    """Test machine lifecycle state transitions."""
    machine = BasicStateMachine()
    assert machine.status == MachineStatus.UNINITIALIZED
    
    # Initialize
    with patch.object(machine, '_initialize_components'), \
         patch.object(machine, '_validate_configuration'):
        machine.initialize()
        assert machine.status == MachineStatus.INITIALIZING
    
    # Activate
    with patch.object(machine, '_start_components'), \
         patch.object(machine, '_start_event_processing'):
        machine.activate()
        assert machine.status == MachineStatus.ACTIVE
    
    # Terminate
    with patch.object(machine, '_stop_event_processing'), \
         patch.object(machine, '_stop_components'), \
         patch.object(machine, '_cleanup_resources'):
        machine.terminate()
        assert machine.status == MachineStatus.TERMINATED


def test_basic_machine_component_coordination():
    """Test coordination between machine components."""
    machine = BasicStateMachine()
    
    # Mock components
    state = Mock(spec=State)
    state.id = "test_state"
    state.initialize = Mock()
    state.activate = Mock()
    state.deactivate = Mock()
    state.enter = Mock()
    state.exit = Mock()
    state.is_valid = Mock(return_value=True)
    
    region = Mock(spec=Region)
    region.id = "test_region"
    region.initialize = Mock()
    region.activate = Mock()
    region.deactivate = Mock()
    region.enter = Mock()
    region.exit = Mock()
    region.is_valid = Mock(return_value=True)
    
    transition = Mock(spec=Transition)
    transition.source = state
    transition.target = None
    transition.event = None
    transition.guard = None
    transition.effect = None
    transition.kind = TransitionKind.INTERNAL
    transition.priority = TransitionPriority.NORMAL
    transition.is_valid = Mock(return_value=True)
    
    event = Mock(spec=Event)
    event.id = "test_event"
    event.kind = EventKind.SIGNAL
    event.priority = EventPriority.NORMAL
    event.data = {}
    event.timeout = None
    event.consumed = False
    event.cancelled = False
    
    # Add components
    machine.add_state(state)
    machine.add_region(region)
    machine.add_transition(transition)
    
    # Mock event queue
    event_queue = Mock(spec=EventQueue)
    event_queue.enqueue = Mock()
    event_queue.start_processing = Mock()
    event_queue.stop_processing = Mock()
    
    # Initialize and activate
    with patch.object(machine, '_event_queue', event_queue):
        machine.initialize()
        machine.activate()
        
        # Process event
        machine.process_event(event)
        
        # Verify component interactions
        state.initialize.assert_called_once()
        state.activate.assert_called_once()
        region.initialize.assert_called_once()
        region.activate.assert_called_once()
        event_queue.enqueue.assert_called_with(event)


def test_basic_machine_dynamic_modification():
    """Test dynamic modifications to machine configuration."""
    machine = BasicStateMachine()
    modifier = MachineModifier()
    
    # Mock components
    state = Mock(spec=State)
    state.id = "test_state"
    state.initialize = Mock()
    state.activate = Mock()
    state.deactivate = Mock()
    state.enter = Mock()
    state.exit = Mock()
    state.is_valid = Mock(return_value=True)
    
    transition = Mock(spec=Transition)
    transition.source = state
    transition.target = None
    transition.event = None
    transition.guard = None
    transition.effect = None
    transition.kind = TransitionKind.INTERNAL
    transition.priority = TransitionPriority.NORMAL
    transition.is_valid = Mock(return_value=True)
    
    # Initialize and activate
    with patch.object(machine, '_event_queue'):
        machine.initialize()
        machine.activate()
    
    # Modify configuration
    with modifier.modify(machine):
        assert machine.status == MachineStatus.MODIFYING
        machine.add_state(state)
        machine.add_transition(transition)
    
    assert machine.status == MachineStatus.ACTIVE
    assert state.id in machine._states
    assert transition in machine._transitions


def test_basic_machine_resource_management():
    """Test machine resource management and cleanup."""
    machine = BasicStateMachine()
    
    # Mock resources
    resource1 = Mock()
    resource2 = Mock()
    
    # Add resources
    machine.add_resource(resource1)
    machine.add_resource(resource2)
    
    # Initialize and activate
    with patch.object(machine, '_event_queue'):
        machine.initialize()
        machine.activate()
    
    # Terminate machine
    machine.terminate()
    
    # Verify resource cleanup
    resource1.cleanup.assert_called_once()
    resource2.cleanup.assert_called_once()
    assert len(machine._resources) == 0
    assert machine.status == MachineStatus.TERMINATED


def test_machine_error_handling():
    """Test error handling during machine operations."""
    machine = BasicStateMachine()
    
    # Test initialization error
    with patch.object(machine, '_initialize_components', side_effect=ValueError("Test error")), \
         pytest.raises(RuntimeError, match="Initialization failed: Test error"):
        machine.initialize()
    assert machine.status == MachineStatus.UNINITIALIZED
    
    # Test activation error
    with patch.object(machine, '_initialize_components'), \
         patch.object(machine, '_validate_configuration'):
        machine.initialize()
    
    with patch.object(machine, '_start_components', side_effect=ValueError("Test error")), \
         pytest.raises(RuntimeError, match="Activation failed: Test error"):
        machine.activate()
    assert machine.status == MachineStatus.INITIALIZING
    
    # Test termination error
    with patch.object(machine, '_start_components'), \
         patch.object(machine, '_start_event_processing'):
        machine.activate()
    
    with patch.object(machine, '_stop_event_processing', side_effect=ValueError("Test error")), \
         pytest.raises(RuntimeError, match="Termination failed: Test error"):
        machine.terminate()
    assert machine.status == MachineStatus.ACTIVE


def test_protocol_machine_initialization():
    """Test protocol machine initialization."""
    machine = ProtocolMachine()
    assert machine.status == MachineStatus.UNINITIALIZED
    assert machine.protocol_name == "default"
    assert len(machine.protocol_rules) == 0


def test_protocol_machine_rule_management():
    """Test protocol rule management."""
    machine = ProtocolMachine()
    
    # Mock components
    state1 = Mock(spec=State)
    state1.id = "state1"
    state1.initialize = Mock()
    state1.activate = Mock()
    state1.is_valid = Mock(return_value=True)
    
    state2 = Mock(spec=State)
    state2.id = "state2"
    state2.initialize = Mock()
    state2.activate = Mock()
    state2.is_valid = Mock(return_value=True)
    
    # Add states
    machine.add_state(state1)
    machine.add_state(state2)
    
    # Add protocol rule
    rule = {
        "operation": "test_op",
        "source": "state1",
        "target": "state2",
        "guard": None,
        "effect": None
    }
    machine.add_protocol_rule(rule)
    
    assert len(machine.protocol_rules) == 1
    assert rule in machine.protocol_rules


def test_protocol_machine_operation_validation():
    """Test protocol operation validation."""
    machine = ProtocolMachine()
    
    # Mock components
    state1 = Mock(spec=State)
    state1.id = "state1"
    state1.initialize = Mock()
    state1.activate = Mock()
    state1.is_valid = Mock(return_value=True)
    
    state2 = Mock(spec=State)
    state2.id = "state2"
    state2.initialize = Mock()
    state2.activate = Mock()
    state2.is_valid = Mock(return_value=True)
    
    # Add states
    machine.add_state(state1)
    machine.add_state(state2)
    
    # Add protocol rule
    rule = {
        "operation": "test_op",
        "source": "state1",
        "target": "state2",
        "guard": None,
        "effect": None
    }
    machine.add_protocol_rule(rule)
    
    # Initialize and activate
    with patch.object(machine, '_event_queue'):
        machine.initialize()
        machine.activate()
    
    # Test valid operation
    event = Mock(spec=Event)
    event.id = "test_event"
    event.kind = EventKind.CALL
    event.priority = EventPriority.NORMAL
    event.data = {
        "operation": "test_op",
        "args": [],
        "kwargs": {}
    }
    event.timeout = None
    event.consumed = False
    event.cancelled = False
    
    machine.process_event(event)
    assert event.consumed
    
    # Test invalid operation
    event2 = Mock(spec=Event)
    event2.id = "test_event2"
    event2.kind = EventKind.CALL
    event2.priority = EventPriority.NORMAL
    event2.data = {
        "operation": "invalid_op",
        "args": [],
        "kwargs": {}
    }
    event2.timeout = None
    event2.consumed = False
    event2.cancelled = False
    
    with pytest.raises(ValueError, match="Invalid operation: invalid_op"):
        machine.process_event(event2)


def test_protocol_machine_sequence_validation():
    """Test protocol operation sequence validation."""
    machine = ProtocolMachine()
    
    # Mock components
    state1 = Mock(spec=State)
    state1.id = "state1"
    state1.initialize = Mock()
    state1.activate = Mock()
    state1.is_valid = Mock(return_value=True)
    
    state2 = Mock(spec=State)
    state2.id = "state2"
    state2.initialize = Mock()
    state2.activate = Mock()
    state2.is_valid = Mock(return_value=True)
    
    state3 = Mock(spec=State)
    state3.id = "state3"
    state3.initialize = Mock()
    state3.activate = Mock()
    state3.is_valid = Mock(return_value=True)
    
    # Add states
    machine.add_state(state1)
    machine.add_state(state2)
    machine.add_state(state3)
    
    # Add protocol rules for sequence: op1 -> op2 -> op3
    rules = [
        {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "guard": None,
            "effect": None
        },
        {
            "operation": "op2",
            "source": "state2",
            "target": "state3",
            "guard": None,
            "effect": None
        }
    ]
    for rule in rules:
        machine.add_protocol_rule(rule)
    
    # Initialize and activate
    with patch.object(machine, '_event_queue'):
        machine.initialize()
        machine.activate()
    
    # Test valid sequence
    event1 = Mock(spec=Event)
    event1.id = "event1"
    event1.kind = EventKind.CALL
    event1.priority = EventPriority.NORMAL
    event1.data = {"operation": "op1", "args": [], "kwargs": {}}
    event1.timeout = None
    event1.consumed = False
    event1.cancelled = False
    
    event2 = Mock(spec=Event)
    event2.id = "event2"
    event2.kind = EventKind.CALL
    event2.priority = EventPriority.NORMAL
    event2.data = {"operation": "op2", "args": [], "kwargs": {}}
    event2.timeout = None
    event2.consumed = False
    event2.cancelled = False
    
    machine.process_event(event1)
    assert event1.consumed
    machine.process_event(event2)
    assert event2.consumed
    
    # Test invalid sequence (op2 before op1)
    machine2 = ProtocolMachine()
    machine2.add_state(state1)
    machine2.add_state(state2)
    machine2.add_state(state3)
    for rule in rules:
        machine2.add_protocol_rule(rule)
    
    with patch.object(machine2, '_event_queue'):
        machine2.initialize()
        machine2.activate()
    
    with pytest.raises(ValueError, match="Invalid operation sequence"):
        machine2.process_event(event2)


def test_submachine_initialization():
    """Test submachine initialization."""
    machine = SubmachineMachine("test_submachine")
    assert machine.status == MachineStatus.UNINITIALIZED
    assert machine.name == "test_submachine"
    assert machine.parent_count == 0


def test_submachine_parent_reference_management():
    """Test submachine parent reference management."""
    submachine = SubmachineMachine("test_submachine")
    parent1 = BasicStateMachine()
    parent2 = BasicStateMachine()
    
    # Add parent references
    submachine.add_parent_reference(parent1)
    submachine.add_parent_reference(parent2)
    assert submachine.parent_count == 2
    
    # Try to add duplicate reference
    with pytest.raises(ValueError, match="Parent machine already referenced"):
        submachine.add_parent_reference(parent1)
    
    # Remove parent reference
    submachine.remove_parent_reference(parent1)
    assert submachine.parent_count == 1
    
    # Try to remove non-existent reference
    with pytest.raises(ValueError, match="Parent machine not referenced"):
        submachine.remove_parent_reference(parent1)
    
    # Try to add/remove None parent
    with pytest.raises(ValueError, match="Parent machine cannot be None"):
        submachine.add_parent_reference(None)
    with pytest.raises(ValueError, match="Parent machine cannot be None"):
        submachine.remove_parent_reference(None)


def test_submachine_data_context():
    """Test submachine data context management."""
    machine = SubmachineMachine("test_submachine")
    
    # Set and get data
    machine.set_data("key1", "value1")
    machine.set_data("key2", 42)
    assert machine.get_data("key1") == "value1"
    assert machine.get_data("key2") == 42
    
    # Try to get non-existent key
    with pytest.raises(KeyError):
        machine.get_data("non_existent")
    
    # Clear data
    machine.clear_data()
    with pytest.raises(KeyError):
        machine.get_data("key1")


def test_submachine_lifecycle():
    """Test submachine lifecycle with parent references and data."""
    submachine = SubmachineMachine("test_submachine")
    parent = BasicStateMachine()
    
    # Add parent and data
    submachine.add_parent_reference(parent)
    submachine.set_data("key", "value")
    
    # Initialize
    with patch.object(submachine, '_initialize_components'), \
         patch.object(submachine, '_validate_configuration'):
        submachine.initialize()
        assert submachine.status == MachineStatus.INITIALIZING
    
    # Activate
    with patch.object(submachine, '_start_components'), \
         patch.object(submachine, '_start_event_processing'):
        submachine.activate()
        assert submachine.status == MachineStatus.ACTIVE
    
    # Terminate
    with patch.object(submachine, '_stop_event_processing'), \
         patch.object(submachine, '_stop_components'), \
         patch.object(submachine, '_cleanup_resources'):
        submachine.terminate()
        assert submachine.status == MachineStatus.TERMINATED
        assert submachine.parent_count == 0
        with pytest.raises(KeyError):
            submachine.get_data("key")


def test_submachine_cyclic_reference_detection():
    """Test detection of cyclic references in submachine configuration."""
    submachine = SubmachineMachine("test_submachine")
    
    # Create a state that references the submachine itself
    state = Mock(spec=State)
    state.id = "test_state"
    state.initialize = Mock()
    state.activate = Mock()
    state.is_valid = Mock(return_value=True)
    state.submachine = submachine
    
    # Add state and try to initialize
    submachine.add_state(state)
    
    with pytest.raises(ValueError, match="Cyclic submachine reference detected"):
        submachine.initialize()


def test_machine_monitor_initialization():
    """Test machine monitor initialization."""
    monitor = MachineMonitor()
    assert monitor.metrics == {}
    assert monitor.history == []
    assert monitor.event_count == 0


def test_machine_monitor_event_tracking():
    """Test machine monitor event tracking."""
    monitor = MachineMonitor()
    machine = Mock(spec=BasicStateMachine)
    
    # Track state change event
    event = {
        'type': 'state_change',
        'machine': machine,
        'from_state': 'state1',
        'to_state': 'state2',
        'timestamp': 123456789
    }
    monitor.track_event(event)
    
    assert monitor.event_count == 1
    assert len(monitor.history) == 1
    assert monitor.history[0] == event
    
    # Track transition event
    event2 = {
        'type': 'transition',
        'machine': machine,
        'transition': 'trans1',
        'timestamp': 123456790
    }
    monitor.track_event(event2)
    
    assert monitor.event_count == 2
    assert len(monitor.history) == 2
    assert monitor.history[1] == event2


def test_machine_monitor_metrics():
    """Test machine monitor metrics tracking."""
    monitor = MachineMonitor()
    machine = Mock(spec=BasicStateMachine)
    
    # Update metrics
    monitor.update_metric('state_changes', 1)
    monitor.update_metric('transitions', 1)
    monitor.update_metric('events_processed', 1)
    
    assert monitor.get_metric('state_changes') == 1
    assert monitor.get_metric('transitions') == 1
    assert monitor.get_metric('events_processed') == 1
    
    # Update existing metrics
    monitor.update_metric('state_changes', 2)
    assert monitor.get_metric('state_changes') == 3  # Cumulative
    
    # Try to get non-existent metric
    with pytest.raises(KeyError):
        monitor.get_metric('non_existent')


def test_machine_monitor_query():
    """Test machine monitor query capabilities."""
    monitor = MachineMonitor()
    machine = Mock(spec=BasicStateMachine)
    
    # Add some events
    events = [
        {
            'type': 'state_change',
            'machine': machine,
            'from_state': 'state1',
            'to_state': 'state2',
            'timestamp': 123456789
        },
        {
            'type': 'transition',
            'machine': machine,
            'transition': 'trans1',
            'timestamp': 123456790
        },
        {
            'type': 'state_change',
            'machine': machine,
            'from_state': 'state2',
            'to_state': 'state3',
            'timestamp': 123456791
        }
    ]
    for event in events:
        monitor.track_event(event)
    
    # Query state changes
    state_changes = monitor.query_events(event_type='state_change')
    assert len(state_changes) == 2
    assert all(e['type'] == 'state_change' for e in state_changes)
    
    # Query transitions
    transitions = monitor.query_events(event_type='transition')
    assert len(transitions) == 1
    assert all(e['type'] == 'transition' for e in transitions)
    
    # Query with time range
    recent_events = monitor.query_events(start_time=123456790)
    assert len(recent_events) == 2
    assert all(e['timestamp'] >= 123456790 for e in recent_events)


def test_machine_monitor_thread_safety():
    """Test machine monitor thread safety."""
    monitor = MachineMonitor()
    
    # Test concurrent metric updates
    with patch.object(monitor, '_metrics_lock') as mock_lock:
        monitor.update_metric('test_metric', 1)
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
    
    # Test concurrent event tracking
    with patch.object(monitor, '_history_lock') as mock_lock:
        monitor.track_event({'type': 'test'})
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


def test_machine_builder_initialization():
    """Test machine builder initialization."""
    builder = MachineBuilder()
    assert builder.machine_type == BasicStateMachine
    assert builder.components == {}
    assert builder.dependencies == {}


def test_machine_builder_configuration():
    """Test machine builder configuration."""
    builder = MachineBuilder()
    
    # Configure machine type
    builder.set_machine_type(ProtocolMachine)
    assert builder.machine_type == ProtocolMachine
    
    # Configure components
    state = Mock(spec=State)
    state.id = "test_state"
    state.initialize = Mock()
    state.activate = Mock()
    state.is_valid = Mock(return_value=True)
    
    region = Mock(spec=Region)
    region.id = "test_region"
    region.initialize = Mock()
    region.activate = Mock()
    region.is_valid = Mock(return_value=True)
    
    transition = Mock(spec=Transition)
    transition.is_valid = Mock(return_value=True)
    
    builder.add_component("states", state)
    builder.add_component("regions", region)
    builder.add_component("transitions", transition)
    
    assert len(builder.components) == 3
    assert builder.components["states"] == [state]
    assert builder.components["regions"] == [region]
    assert builder.components["transitions"] == [transition]


def test_machine_builder_validation():
    """Test machine builder validation."""
    builder = MachineBuilder()
    
    # Add invalid state
    state = Mock(spec=State)
    state.id = "test_state"
    state.initialize = Mock()
    state.activate = Mock()
    state.is_valid = Mock(return_value=False)
    
    builder.add_component("states", state)
    
    with pytest.raises(ValueError, match="Invalid state configuration"):
        builder.build()
    
    # Add invalid region
    region = Mock(spec=Region)
    region.id = "test_region"
    region.initialize = Mock()
    region.activate = Mock()
    region.is_valid = Mock(return_value=False)
    
    builder.add_component("regions", region)
    
    with pytest.raises(ValueError, match="Invalid region configuration"):
        builder.build()
    
    # Add invalid transition
    transition = Mock(spec=Transition)
    transition.is_valid = Mock(return_value=False)
    
    builder.add_component("transitions", transition)
    
    with pytest.raises(ValueError, match="Invalid transition configuration"):
        builder.build()


def test_machine_builder_dependency_tracking():
    """Test machine builder dependency tracking."""
    builder = MachineBuilder()
    
    # Create components with dependencies
    state1 = Mock(spec=State)
    state1.id = "state1"
    state1.initialize = Mock()
    state1.activate = Mock()
    state1.is_valid = Mock(return_value=True)
    
    state2 = Mock(spec=State)
    state2.id = "state2"
    state2.initialize = Mock()
    state2.activate = Mock()
    state2.is_valid = Mock(return_value=True)
    
    transition = Mock(spec=Transition)
    transition.source_state = "state1"
    transition.target_state = "state2"
    transition.is_valid = Mock(return_value=True)
    
    # Add components
    builder.add_component("states", state1)
    builder.add_component("states", state2)
    builder.add_component("transitions", transition)
    
    # Add dependency
    builder.add_dependency("transitions", "states")
    
    # Build should succeed (valid dependencies)
    machine = builder.build()
    assert isinstance(machine, BasicStateMachine)
    
    # Test invalid dependency
    builder2 = MachineBuilder()
    builder2.add_component("transitions", transition)
    builder2.add_dependency("transitions", "states")
    
    with pytest.raises(ValueError, match="Unresolved dependency: states required by transitions"):
        builder2.build()


def test_machine_builder_thread_safety():
    """Test machine builder thread safety."""
    builder = MachineBuilder()
    
    # Test concurrent component addition
    with patch.object(builder, '_component_lock') as mock_lock:
        state = Mock(spec=State)
        builder.add_component("states", state)
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()
    
    # Test concurrent dependency addition
    with patch.object(builder, '_dependency_lock') as mock_lock:
        builder.add_dependency("transitions", "states")
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once() 