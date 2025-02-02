"""
State machine orchestration and lifecycle management.

Architecture:
- Orchestrates state machine components
- Manages machine lifecycle and configuration
- Coordinates core component interactions
- Integrates with Monitor for introspection
- Handles dynamic modifications

Design Patterns:
- Facade Pattern: Component coordination
- Builder Pattern: Machine configuration
- Observer Pattern: State notifications
- Mediator Pattern: Component interaction
- Strategy Pattern: Machine policies

Responsibilities:
1. Machine Lifecycle
   - Initialization
   - Configuration
   - Dynamic modification
   - Version management
   - Termination

2. Component Coordination
   - State management
   - Transition handling
   - Event processing
   - Region execution
   - Resource control

3. Machine Configuration
   - Validation rules
   - Security policies
   - Resource limits
   - Extension settings
   - Monitoring options

4. Dynamic Modifications
   - Runtime changes
   - Semantic consistency
   - State preservation
   - Version compatibility
   - Modification atomicity

Security:
- Configuration validation
- Component isolation
- Resource management
- Extension control

Cross-cutting:
- Error handling
- Performance monitoring
- Machine metrics
- Thread safety

Dependencies:
- state.py: State management
- transition.py: Transition handling
- event.py: Event processing
- region.py: Region coordination
- monitor.py: Machine monitoring
"""

import threading
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from threading import Event, RLock
from typing import Any, Dict, List, Optional, Set, Type, Callable
from weakref import ref
import time
import copy

from gotstate.core.event import Event, EventKind, EventQueue
from gotstate.core.region import Region
from gotstate.core.state import State
from gotstate.core.transition import Transition


class MachineStatus(Enum):
    """Defines the possible states of a state machine.

    Used to track machine lifecycle and coordinate operations.
    """

    UNINITIALIZED = auto()  # Machine not yet configured
    INITIALIZING = auto()  # Machine being configured
    ACTIVE = auto()  # Machine running normally
    MODIFYING = auto()  # Machine being modified
    TERMINATING = auto()  # Machine shutting down
    TERMINATED = auto()  # Machine fully stopped


class StateMachine:
    """Represents a hierarchical state machine.

    The StateMachine class implements the Facade pattern to coordinate
    all components and manage the machine lifecycle.

    Class Invariants:
    1. Must maintain semantic consistency
    2. Must preserve state hierarchy
    3. Must coordinate components
    4. Must enforce configuration
    5. Must handle modifications
    6. Must manage resources
    7. Must track versions
    8. Must support introspection
    9. Must isolate extensions
    10. Must maintain metrics

    Design Patterns:
    - Facade: Coordinates components
    - Builder: Configures machine
    - Observer: Notifies of changes
    - Mediator: Manages interactions
    - Strategy: Implements policies
    - Command: Encapsulates operations

    Data Structures:
    - Tree for state hierarchy
    - Graph for transitions
    - Queue for events
    - Map for components
    - Set for active states

    Algorithms:
    - Configuration validation
    - Version compatibility
    - Resource allocation
    - Modification planning
    - Metrics aggregation

    Threading/Concurrency Guarantees:
    1. Thread-safe operations
    2. Atomic modifications
    3. Synchronized components
    4. Safe concurrent access
    5. Lock-free inspection
    6. Mutex protection

    Performance Characteristics:
    1. O(1) status checks
    2. O(log n) component access
    3. O(h) hierarchy operations where h is height
    4. O(m) modifications where m is change count
    5. O(v) version checks where v is version count

    Resource Management:
    1. Bounded memory usage
    2. Controlled thread allocation
    3. Resource pooling
    4. Automatic cleanup
    5. Load distribution
    """

    def __init__(self):
        """Initialize a new state machine instance.

        The machine starts in UNINITIALIZED status and must be explicitly
        initialized before use.

        Thread Safety:
        - All status transitions are atomic
        - Component collections are thread-safe
        - Resource management is synchronized

        Raises:
            NotImplementedError: If called on base class
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    @property
    def status(self) -> MachineStatus:
        """Get the current machine status."""
        raise NotImplementedError("StateMachine is an abstract base class")

    def initialize(self) -> None:
        """Initialize the state machine.

        Transitions from UNINITIALIZED to INITIALIZING status.
        Validates configuration and prepares components.

        Raises:
            NotImplementedError: If called on base class
            ValueError: If machine is not in UNINITIALIZED status
            RuntimeError: If initialization fails
        """
        with self._status_lock:
            if self._status != MachineStatus.UNINITIALIZED:
                raise ValueError("Machine must be in UNINITIALIZED status")
            self._status = MachineStatus.INITIALIZING

            try:
                try:
                    # Initialize components
                    self._initialize_components()
                except Exception as e:
                    raise RuntimeError(f"Initialization failed: {e}")

                # Validate configuration
                self._validate_configuration()

            except Exception as e:
                # Rollback on failure
                self._cleanup_components()
                self._status = MachineStatus.UNINITIALIZED
                raise e

    def activate(self) -> None:
        """Activate the state machine.

        Transitions from INITIALIZING to ACTIVE status.
        Starts event processing and component execution.

        Raises:
            NotImplementedError: If called on base class
            ValueError: If machine is not in INITIALIZING status
            RuntimeError: If activation fails
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def terminate(self) -> None:
        """Terminate the state machine.

        Transitions through TERMINATING to TERMINATED status.
        Stops all processing and cleans up resources.

        Raises:
            NotImplementedError: If called on base class
            RuntimeError: If termination fails
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def add_state(self, state: State) -> None:
        """Add a state to the machine.

        Args:
            state: The state to add

        Raises:
            NotImplementedError: If called on base class
            ValueError: If state ID conflicts or machine not modifiable
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def add_region(self, region: Region) -> None:
        """Add a region to the machine.

        Args:
            region: The region to add

        Raises:
            NotImplementedError: If called on base class
            ValueError: If region ID conflicts or machine not modifiable
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the machine.

        Args:
            transition: The transition to add

        Raises:
            NotImplementedError: If called on base class
            ValueError: If machine not modifiable
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def add_resource(self, resource: Any) -> None:
        """Add a resource to be managed by the machine.

        Args:
            resource: The resource to manage

        Raises:
            NotImplementedError: If called on base class
            ValueError: If machine not modifiable
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def process_event(self, event: Event) -> None:
        """Process an event through the state machine.

        Args:
            event: The event to process

        Raises:
            NotImplementedError: If called on base class
            RuntimeError: If machine not active
        """
        raise NotImplementedError("StateMachine is an abstract base class")

    def _initialize_components(self) -> None:
        """Initialize all machine components.

        Called during machine initialization.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _validate_configuration(self) -> None:
        """Validate machine configuration.

        Called during machine initialization.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _start_components(self) -> None:
        """Start all machine components.

        Called during machine activation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _stop_components(self) -> None:
        """Stop all machine components.

        Called during machine termination.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _cleanup_components(self) -> None:
        """Clean up machine components.

        Called during initialization failure.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _start_event_processing(self) -> None:
        """Start event processing.

        Called during machine activation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _stop_event_processing(self) -> None:
        """Stop event processing.

        Called during machine termination.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _cleanup_resources(self) -> None:
        """Clean up managed resources.

        Called during machine termination.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class BasicStateMachine(StateMachine):
    """Basic implementation of a hierarchical state machine.

    Provides concrete implementation of the StateMachine interface
    with standard lifecycle and component management.

    Class Invariants:
    1. Must maintain semantic consistency
    2. Must preserve state hierarchy
    3. Must coordinate components
    4. Must enforce configuration
    5. Must handle modifications
    6. Must manage resources
    7. Must track versions
    8. Must support introspection
    9. Must isolate extensions
    10. Must maintain metrics
    """

    def __init__(self):
        """Initialize a new state machine instance."""
        # Machine status and synchronization
        self._status = MachineStatus.UNINITIALIZED
        self._status_lock = RLock()

        # Component collections with thread-safe access
        self._states: Dict[str, State] = {}
        self._regions: Dict[str, Region] = {}
        self._transitions: Set[Transition] = set()
        self._resources: Set[Any] = set()
        self._collection_lock = RLock()

        # Event processing
        self._event_queue = EventQueue()
        self._processing_lock = RLock()

        # Monitoring and metrics
        self._monitor = MachineMonitor()
        self._monitor_lock = RLock()

        # Version management
        self._version = "1.0.0"
        self._version_lock = RLock()

        # Security and resources
        self._security_policies: Dict[str, Callable[[str], bool]] = {}
        self._resource_limits: Dict[str, int] = {
            "max_states": 1000,
            "max_regions": 100,
            "max_transitions": 5000,
            "max_events_queued": 10000,
            "max_resources": 100
        }
        self._security_lock = RLock()

    @property
    def status(self) -> MachineStatus:
        """Get the current machine status."""
        with self._status_lock:
            return self._status

    def initialize(self) -> None:
        """Initialize the state machine."""
        with self._status_lock:
            if self._status != MachineStatus.UNINITIALIZED:
                raise ValueError("Machine must be in UNINITIALIZED status")
            self._status = MachineStatus.INITIALIZING

            try:
                try:
                    # Initialize components
                    self._initialize_components()
                except Exception as e:
                    raise RuntimeError(f"Initialization failed: {e}")

                # Validate configuration
                self._validate_configuration()

            except Exception as e:
                # Rollback on failure
                self._cleanup_components()
                self._status = MachineStatus.UNINITIALIZED
                raise e

    def activate(self) -> None:
        """Activate the state machine.

        Transitions from INITIALIZING to ACTIVE status.
        Starts event processing and component execution.

        Raises:
            NotImplementedError: If called on base class
            ValueError: If machine is not in INITIALIZING status
            RuntimeError: If activation fails
        """
        with self._status_lock:
            if self._status != MachineStatus.INITIALIZING:
                raise ValueError("Machine must be in INITIALIZING status")
            self._status = MachineStatus.ACTIVE

            try:
                # Start components
                self._start_components()

                # Begin event processing
                self._start_event_processing()

            except Exception as e:
                # Rollback on failure
                self._stop_components()
                self._status = MachineStatus.INITIALIZING
                raise RuntimeError(f"Activation failed: {e}")

    def terminate(self) -> None:
        """Terminate the state machine.

        Transitions through TERMINATING to TERMINATED status.
        Stops all processing and cleans up resources.

        Raises:
            NotImplementedError: If called on base class
            RuntimeError: If termination fails
        """
        with self._status_lock:
            if self._status == MachineStatus.TERMINATED:
                return

            prev_status = self._status
            self._status = MachineStatus.TERMINATING

            try:
                # Stop processing
                self._stop_event_processing()

                # Stop components
                self._stop_components()

                # Cleanup resources
                self._cleanup_resources()

                self._status = MachineStatus.TERMINATED

            except Exception as e:
                # Attempt to restore previous status
                self._status = prev_status
                raise RuntimeError(f"Termination failed: {e}")

    def add_state(self, state: State) -> None:
        """Add a state to the machine."""
        with self._collection_lock:
            if self.status == MachineStatus.ACTIVE:
                raise ValueError("Cannot add states while machine is active")
            if state is None:
                raise ValueError("State cannot be None")
            if not isinstance(state, State):
                raise ValueError("Component must be a State instance")
            if state.id in self._states:
                raise ValueError(f"State ID '{state.id}' already exists")
            self._states[state.id] = state

    def add_region(self, region: Region) -> None:
        """Add a region to the machine."""
        with self._collection_lock:
            if self.status == MachineStatus.ACTIVE:
                raise ValueError("Cannot add regions while machine is active")
            if region is None:
                raise ValueError("Region cannot be None")
            if not isinstance(region, Region):
                raise ValueError("Component must be a Region instance")
            if region.id in self._regions:
                raise ValueError(f"Region ID '{region.id}' already exists")
            self._regions[region.id] = region

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the machine."""
        with self._collection_lock:
            if self.status == MachineStatus.ACTIVE:
                raise ValueError("Cannot add transitions while machine is active")
            self._transitions.add(transition)

    def add_resource(self, resource: Any) -> None:
        """Add a resource to be managed by the machine."""
        with self._collection_lock:
            if self.status == MachineStatus.ACTIVE:
                raise ValueError("Cannot add resources while machine is active")
            if resource is None:
                raise ValueError("Resource cannot be None")
            self._resources.add(resource)

    def process_event(self, event: Event) -> None:
        """Process an event through the state machine."""
        if self.status != MachineStatus.ACTIVE:
            raise RuntimeError("Machine must be ACTIVE to process events")

        with self._processing_lock:
            try:
                self._track_event("event_processing", {
                    "event_id": id(event),
                    "event_kind": event.kind.name,
                    "event_data": event.data
                })
                self._event_queue.enqueue(event)
                self._track_event("event_queued", {
                    "event_id": id(event),
                    "queue_size": len(self._event_queue)
                })
            except Exception as e:
                self._track_event("event_processing_failure", {
                    "event_id": id(event),
                    "error": str(e)
                })
                raise

    def _track_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Track a machine event with monitoring.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        with self._monitor_lock:
            event_data = {
                "type": event_type,
                "timestamp": time.time(),
                "machine_status": self.status.name,
                **details
            }
            self._monitor.track_event(event_data)
            
    def _initialize_components(self) -> None:
        """Initialize all machine components."""
        with self._collection_lock:
            # Initialize states
            for state in self._states.values():
                try:
                    self._track_event("component_init", {
                        "component_type": "state",
                        "component_id": state.id
                    })
                    state.initialize()
                    self._track_event("component_init_success", {
                        "component_type": "state",
                        "component_id": state.id
                    })
                except Exception as e:
                    self._track_event("component_init_failure", {
                        "component_type": "state",
                        "component_id": state.id,
                        "error": str(e)
                    })
                    raise

            # Initialize regions
            for region in self._regions.values():
                try:
                    self._track_event("component_init", {
                        "component_type": "region",
                        "component_id": region.id
                    })
                    region.initialize()
                    self._track_event("component_init_success", {
                        "component_type": "region",
                        "component_id": region.id
                    })
                except Exception as e:
                    self._track_event("component_init_failure", {
                        "component_type": "region",
                        "component_id": region.id,
                        "error": str(e)
                    })
                    raise

    def _validate_configuration(self) -> None:
        """Validate machine configuration."""
        with self._collection_lock:
            # Check resource limits first
            if not self.check_resource_limits():
                raise ValueError("Resource limits exceeded")

            # Validate states
            for state in self._states.values():
                if not state.is_valid():
                    raise ValueError(f"Invalid state configuration: {state.id}")

            # Validate regions
            for region in self._regions.values():
                if not region.is_valid():
                    raise ValueError(f"Invalid region configuration: {region.id}")

            # Validate transitions
            for transition in self._transitions:
                if not transition.is_valid():
                    raise ValueError("Invalid transition configuration")

    def _start_components(self) -> None:
        """Start all machine components."""
        with self._collection_lock:
            try:
                # Allocate resources first
                for resource in self._resources:
                    resource.allocate()

                # Activate states
                for state in self._states.values():
                    state.enter()

                # Activate regions
                for region in self._regions.values():
                    region.activate()
            except Exception as e:
                # Clean up on failure
                for resource in self._resources:
                    try:
                        resource.cleanup()
                    except Exception:
                        pass  # Best effort cleanup
                raise e

    def _stop_components(self) -> None:
        """Stop all machine components."""
        with self._collection_lock:
            # Deactivate states
            for state in self._states.values():
                state.exit()

            # Deactivate regions
            for region in self._regions.values():
                region.deactivate()

    def _cleanup_components(self) -> None:
        """Clean up machine components."""
        with self._collection_lock:
            self._states.clear()
            self._regions.clear()
            self._transitions.clear()

    def _start_event_processing(self) -> None:
        """Start event processing."""
        with self._processing_lock:
            self._event_queue.start_processing()

    def _stop_event_processing(self) -> None:
        """Stop event processing."""
        with self._processing_lock:
            self._event_queue.stop_processing()

    def _cleanup_resources(self) -> None:
        """Clean up managed resources."""
        with self._collection_lock:
            for resource in self._resources:
                if hasattr(resource, "cleanup"):
                    resource.cleanup()
            self._resources.clear()

    def get_version(self) -> str:
        """Get the current machine version.

        Returns:
            The version string
        """
        with self._version_lock:
            return self._version

    def validate_version_compatibility(self, other_version: str) -> bool:
        """Check version compatibility.

        Args:
            other_version: Version to check compatibility with

        Returns:
            True if versions are compatible
        """
        with self._version_lock:
            # Simple semantic version comparison
            current_parts = [int(x) for x in self._version.split(".")]
            other_parts = [int(x) for x in other_version.split(".")]
            
            # Major version must match
            return current_parts[0] == other_parts[0]

    def add_security_policy(self, operation: str, validator: Callable[[str], bool]) -> None:
        """Add a security policy for an operation.

        Args:
            operation: Operation to secure
            validator: Validation function
        """
        with self._security_lock:
            self._security_policies[operation] = validator
            self._track_event("security_policy_added", {
                "operation": operation
            })

    def validate_security_policy(self, operation: str) -> bool:
        """Validate operation against security policy.

        Args:
            operation: Operation to validate

        Returns:
            True if operation is allowed
        """
        with self._security_lock:
            # If no policy exists for this operation, it's not allowed
            if operation not in self._security_policies:
                return False
            
            # Validate using the operation's specific policy
            try:
                return self._security_policies[operation](operation)
            except Exception:
                return False

    def set_resource_limit(self, resource: str, limit: int) -> None:
        """Set a resource usage limit.

        Args:
            resource: Resource type
            limit: Maximum allowed value
        """
        with self._security_lock:
            self._resource_limits[resource] = limit
            self._track_event("resource_limit_set", {
                "resource": resource,
                "limit": limit
            })

    def check_resource_limits(self) -> bool:
        """Check if resource usage is within limits.

        Returns:
            True if within limits
        """
        with self._collection_lock, self._security_lock:
            if len(self._states) > self._resource_limits["max_states"]:
                return False
            if len(self._regions) > self._resource_limits["max_regions"]:
                return False
            if len(self._transitions) > self._resource_limits["max_transitions"]:
                return False
            if len(self._resources) > self._resource_limits["max_resources"]:
                return False
            if len(self._event_queue) > self._resource_limits["max_events_queued"]:
                return False
            return True


class ProtocolMachine(BasicStateMachine):
    """Represents a protocol state machine.

    ProtocolMachine enforces protocol constraints and operation
    sequences for behavioral specifications.

    Class Invariants:
    1. Must enforce protocol rules
    2. Must validate operations
    3. Must track protocol state
    4. Must maintain sequences

    Design Patterns:
    - State: Manages protocol states
    - Strategy: Implements protocols
    - Command: Encapsulates operations

    Threading/Concurrency Guarantees:
    1. Thread-safe validation
    2. Atomic operations
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) state checks
    2. O(r) rule validation where r is rule count
    3. O(s) sequence tracking where s is sequence length
    """

    def __init__(self, protocol_name: str = "default"):
        """Initialize a protocol state machine.

        Args:
            protocol_name: Name of the protocol
        """
        super().__init__()
        self._protocol_name = protocol_name
        self._protocol_rules: List[Dict[str, Any]] = []
        self._current_state: Optional[State] = None
        self._rule_lock = RLock()
        self._operation_sequence: List[str] = []
        self._sequence_lock = RLock()
        self._sequence_rules: List[Dict[str, List[str]]] = []

    @property
    def protocol_name(self) -> str:
        """Get the protocol name."""
        return self._protocol_name

    @property
    def protocol_rules(self) -> List[Dict[str, Any]]:
        """Get the protocol rules."""
        with self._rule_lock:
            return self._protocol_rules.copy()

    def add_protocol_rule(self, rule: Dict[str, Any]) -> None:
        """Add a protocol rule.

        Args:
            rule: Protocol rule specification

        Raises:
            ValueError: If rule is invalid or conflicts with existing rules
        """
        if not isinstance(rule, dict):
            raise ValueError("Rule must be a dictionary")

        required_fields = {"operation", "source", "target"}
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Rule must contain fields: {required_fields}")

        with self._rule_lock:
            # Validate rule against existing rules
            for existing_rule in self._protocol_rules:
                if existing_rule["operation"] == rule["operation"] and existing_rule["source"] == rule["source"]:
                    raise ValueError(f"Rule conflict: {rule['operation']} from {rule['source']}")

            self._protocol_rules.append(rule.copy())

    def add_sequence_rule(self, sequence_rule: Dict[str, List[str]]) -> None:
        """Add a sequence validation rule.

        Args:
            sequence_rule: Dictionary mapping operation to allowed next operations

        Raises:
            ValueError: If rule is invalid
        """
        if not isinstance(sequence_rule, dict):
            raise ValueError("Sequence rule must be a dictionary")
        
        with self._rule_lock:
            self._sequence_rules.append(sequence_rule.copy())
            self._track_event("sequence_rule_added", {
                "rule": sequence_rule
            })

    def _validate_sequence(self, operation: str) -> bool:
        """Validate if an operation is allowed in the current sequence.

        Args:
            operation: Operation to validate

        Returns:
            bool: True if operation is valid in current sequence
        """
        with self._sequence_lock:
            if not self._operation_sequence:
                # First operation is always allowed
                return True

            last_operation = self._operation_sequence[-1]
            
            # Check all sequence rules
            for rule in self._sequence_rules:
                if last_operation in rule:
                    allowed_next = rule[last_operation]
                    if operation not in allowed_next:
                        return False

            return True

    def _validate_operation(self, operation: str) -> bool:
        """Validate if an operation is allowed.

        Args:
            operation: Operation to validate

        Returns:
            bool: True if operation is valid
        """
        if not self._current_state:
            return False

        # First validate sequence
        if not self._validate_sequence(operation):
            self._track_event("sequence_validation_failed", {
                "operation": operation,
                "current_sequence": self._operation_sequence.copy()
            })
            return False

        # Then validate state transition
        with self._rule_lock:
            for rule in self._protocol_rules:
                if (rule["operation"] == operation and 
                    rule["source"] == self._current_state.id):
                    return True

        self._track_event("state_validation_failed", {
            "operation": operation,
            "current_state": self._current_state.id
        })
        return False

    def _apply_operation(self, operation: str) -> None:
        """Apply a protocol operation."""
        if not self._current_state:
            raise ValueError("No current state")

        # Find matching rule
        rule = None
        with self._rule_lock:
            for r in self._protocol_rules:
                if (r["operation"] == operation and 
                    r["source"] == self._current_state.id):
                    rule = r
                    break

        if not rule:
            raise ValueError(f"No rule found for operation {operation} in state {self._current_state.id}")

        # Verify target state exists
        target_state = self._states.get(rule["target"])
        if not target_state:
            raise ValueError(f"Target state not found: {rule['target']}")

        # Evaluate guard condition
        guard = rule.get("guard")
        if guard and not guard():
            self._track_event("guard_condition_failed", {
                "operation": operation,
                "state": self._current_state.id
            })
            raise ValueError("Guard condition failed")

        try:
            # Execute effect
            effect = rule.get("effect")
            if effect:
                effect()

            # Update current state
            self._current_state = target_state

            # Record operation in sequence
            with self._sequence_lock:
                self._operation_sequence.append(operation)

            self._track_event("operation_applied", {
                "operation": operation,
                "from_state": rule["source"],
                "to_state": rule["target"]
            })

        except Exception as e:
            self._track_event("operation_failed", {
                "operation": operation,
                "error": str(e)
            })
            raise

    def process_event(self, event: Event) -> None:
        """Process an event, validating protocol constraints."""
        if event.kind != EventKind.CALL:
            super().process_event(event)
            return

        operation = event.data.get("operation")
        if not operation:
            raise ValueError("Call event must specify operation")

        if not self._validate_operation(operation):
            # For sequence validation, we need to check if we're in the initial state
            # If we are, or if we have no current state, it's a sequence error
            if not self._current_state or self._current_state == next(iter(self._states.values())):
                raise ValueError("Invalid operation sequence")
            raise ValueError(f"Invalid operation: {operation}")

        self._apply_operation(operation)
        event.consumed = True

    def initialize(self) -> None:
        """Initialize the protocol machine.

        Raises:
            RuntimeError: If initialization fails
        """
        super().initialize()

        # Set initial state
        if self._states:
            self._current_state = next(iter(self._states.values()))
            self._current_state.enter()

    def terminate(self) -> None:
        """Terminate the protocol machine."""
        if self._current_state:
            self._current_state.exit()
            self._current_state = None
        super().terminate()

    def clear_sequence(self) -> None:
        """Clear the current operation sequence."""
        with self._sequence_lock:
            self._operation_sequence.clear()
            self._track_event("sequence_cleared", {})


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
            self._track_event("data_modified", {
                "key": key,
                "operation": "set"
            })

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
            self._track_event("data_snapshot_created", {
                "snapshot_id": len(self._data_snapshots) - 1
            })

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
            self._track_event("data_snapshot_restored", {
                "snapshot_id": index
            })

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
            self._track_event("data_modified", {
                "key": key,
                "operation": "increment",
                "increment": increment
            })


class MachineBuilder:
    """Builds state machine configurations.

    MachineBuilder implements the Builder pattern to construct
    valid state machine configurations.

    Class Invariants:
    1. Must validate configuration
    2. Must enforce constraints
    3. Must maintain consistency
    4. Must track dependencies

    Design Patterns:
    - Builder: Constructs machines
    - Factory: Creates components
    - Validator: Checks configuration

    Threading/Concurrency Guarantees:
    1. Thread-safe construction
    2. Atomic validation
    3. Safe concurrent access

    Performance Characteristics:
    1. O(c) configuration where c is component count
    2. O(v) validation where v is rule count
    3. O(d) dependency resolution where d is dependency count
    """

    def __init__(self):
        """Initialize the machine builder."""
        self._machine_type = BasicStateMachine
        self._components: Dict[str, List[Any]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._component_lock = threading.Lock()
        self._dependency_lock = threading.Lock()

    @property
    def machine_type(self) -> type:
        """Get the machine type.

        Returns:
            The machine type class
        """
        return self._machine_type

    @property
    def components(self) -> Dict[str, List[Any]]:
        """Get the component collections.

        Returns:
            Dictionary mapping component types to lists of components
        """
        with self._component_lock:
            return {k: list(v) for k, v in self._components.items()}

    @property
    def dependencies(self) -> Dict[str, Set[str]]:
        """Get the component dependencies.

        Returns:
            Dictionary mapping component types to their dependencies
        """
        with self._dependency_lock:
            return {k: set(v) for k, v in self._dependencies.items()}

    def set_machine_type(self, machine_type: Type[StateMachine]) -> None:
        """Set the type of machine to build.

        Args:
            machine_type: The machine type class

        Raises:
            ValueError: If machine_type is not a StateMachine subclass
        """
        if not issubclass(machine_type, StateMachine):
            raise ValueError("Machine type must be a StateMachine subclass")
        self._machine_type = machine_type

    def add_component(self, component_type: str, component: Any) -> None:
        """Add a component to the configuration.

        Args:
            component_type: Type of component (e.g. "states", "regions")
            component: The component to add
        """
        with self._component_lock:
            if component_type not in self._components:
                self._components[component_type] = []
            self._components[component_type].append(component)

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Add a component dependency.

        Args:
            dependent: The dependent component type
            dependency: The required component type

        Raises:
            ValueError: If dependency would create a cycle
        """
        with self._dependency_lock:
            # Check for cyclic dependencies
            if self._would_create_cycle(dependent, dependency):
                raise ValueError(f"Cyclic dependency detected: {dependent} -> {dependency}")

            if dependent not in self._dependencies:
                self._dependencies[dependent] = set()
            self._dependencies[dependent].add(dependency)

    def _would_create_cycle(self, dependent: str, dependency: str) -> bool:
        """Check if adding a dependency would create a cycle.

        Args:
            dependent: The dependent component type
            dependency: The required component type

        Returns:
            True if adding the dependency would create a cycle
        """
        # If the dependency already depends on the dependent, it would create a cycle
        if dependency in self._dependencies and dependent in self._dependencies[dependency]:
            return True

        # Check if there's a path from dependency back to dependent
        visited = set()

        def has_path(start: str, target: str) -> bool:
            if start == target:
                return True
            if start in visited:
                return False
            visited.add(start)
            if start in self._dependencies:
                for next_dep in self._dependencies[start]:
                    if has_path(next_dep, target):
                        return True
            return False

        return has_path(dependency, dependent)

    def build(self) -> StateMachine:
        """Build and validate a machine instance.

        Returns:
            The constructed state machine

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate machine type
        if not self._machine_type:
            raise ValueError("Machine type not set")

        # Define validation order and required types
        validation_order = ["states", "regions", "transitions"]
        required_types = ["states", "regions"]  # Transitions are optional

        # Validate required components
        with self._component_lock:
            # Check if any components are added
            if not self._components:
                raise ValueError("No components added to machine")

            # Check if all required component types are present
            missing_types = [t for t in required_types if t not in self._components]
            if missing_types:
                raise ValueError(f"Missing required component types: {', '.join(missing_types)}")

        # Validate components in specific order
        with self._component_lock:
            for component_type in validation_order:
                if component_type not in self._components:
                    continue

                # Remove trailing 's' from component type for error message
                type_name = component_type[:-1] if component_type.endswith("s") else component_type

                # Validate all components of this type
                for component in self._components[component_type]:
                    if hasattr(component, "is_valid") and not component.is_valid():
                        # Clear components of this type to prevent re-validation
                        self._components[component_type] = []
                        raise ValueError(f"Invalid {type_name} configuration")

        # Validate dependencies
        with self._dependency_lock, self._component_lock:
            for dependent, dependencies in self._dependencies.items():
                for dependency in dependencies:
                    # Check if the dependency exists in any component type
                    dependency_found = False
                    for components in self._components.values():
                        for component in components:
                            if hasattr(component, "id") and component.id == dependency:
                                dependency_found = True
                                break
                        if dependency_found:
                            break
                    if not dependency_found:
                        raise ValueError(f"Unresolved dependency: {dependency} required by {dependent}")

        # Create machine instance
        machine = self._machine_type()

        # Add components in specific order
        with self._component_lock:
            for component_type in validation_order:
                if component_type not in self._components:
                    continue
                add_method = getattr(machine, f"add_{component_type[:-1]}", None)
                if add_method:
                    for component in self._components[component_type]:
                        add_method(component)

        return machine


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
            'states': copy.deepcopy(self._machine._states),
            'regions': copy.deepcopy(self._machine._regions),
            'transitions': copy.deepcopy(self._machine._transitions),
            'resources': copy.deepcopy(self._machine._resources),
            'status': self._machine.status
        }
        
        if isinstance(self._machine, SubmachineMachine):
            snapshot['data_context'] = copy.deepcopy(self._machine._data_context)
            
        return snapshot

    def _restore_snapshot(self) -> None:
        """Restore machine state from snapshot."""
        if not self._snapshot:
            return
            
        self._machine._states = copy.deepcopy(self._snapshot['states'])
        self._machine._regions = copy.deepcopy(self._snapshot['regions'])
        self._machine._transitions = copy.deepcopy(self._snapshot['transitions'])
        self._machine._resources = copy.deepcopy(self._snapshot['resources'])
        
        if isinstance(self._machine, SubmachineMachine) and 'data_context' in self._snapshot:
            self._machine._data_context = copy.deepcopy(self._snapshot['data_context'])

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
                self._machine._track_event("modification_started", {
                    "prev_status": self._prev_status.name
                })
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
                    self._machine._track_event("modification_failed", {
                        "error": str(exc_val)
                    })
                else:
                    # Successful modification
                    self._machine._track_event("modification_completed", {
                        "new_status": self._prev_status.name
                    })
                
                self._machine._status = self._prev_status
        finally:
            # Clear state and release lock
            self._machine = None
            self._prev_status = None
            self._snapshot = None
            self._modification_lock.release()


class MachineMonitor:
    """Monitors state machine execution.

    MachineMonitor implements introspection capabilities and
    collects runtime metrics.

    Class Invariants:
    1. Must track metrics
    2. Must handle events
    3. Must maintain history
    4. Must support queries

    Design Patterns:
    - Observer: Monitors changes
    - Strategy: Implements policies
    - Command: Encapsulates queries

    Threading/Concurrency Guarantees:
    1. Thread-safe monitoring
    2. Atomic updates
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) metric updates
    2. O(q) query execution where q is query complexity
    3. O(h) history tracking where h is history size
    """

    def __init__(self):
        """Initialize the machine monitor."""
        self._metrics = {}
        self._history = []
        self._event_count = 0
        self._metrics_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._event_index = {}  # Type -> List[event indices]
        self._time_index = []   # List of (timestamp, index) tuples

    @property
    def metrics(self) -> Dict[str, int]:
        """Get the current metrics.

        Returns:
            A copy of the metrics dictionary
        """
        with self._metrics_lock:
            return self._metrics.copy()

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get the event history.

        Returns:
            A copy of the event history list
        """
        with self._history_lock:
            return self._history.copy()

    @property
    def event_count(self) -> int:
        """Get the total number of events tracked.

        Returns:
            The event count
        """
        with self._history_lock:
            return self._event_count

    def track_event(self, event: Dict[str, Any]) -> None:
        """Track a machine event.

        Args:
            event: Event data dictionary
        """
        with self._history_lock:
            # Add event to history
            event_index = len(self._history)
            self._history.append(event.copy())
            self._event_count += 1

            # Update type index
            event_type = event.get('type')
            if event_type:
                if event_type not in self._event_index:
                    self._event_index[event_type] = []
                self._event_index[event_type].append(event_index)

            # Update time index
            timestamp = event.get('timestamp')
            if timestamp is not None:
                self._time_index.append((timestamp, event_index))
                # Keep time index sorted
                self._time_index.sort(key=lambda x: x[0])

    def update_metric(self, name: str, value: int) -> None:
        """Update a metric value.

        Args:
            name: Metric name
            value: Value to add to metric
        """
        with self._metrics_lock:
            self._metrics[name] = self._metrics.get(name, 0) + value

    def get_metric(self, name: str) -> int:
        """Get a metric value.

        Args:
            name: Metric name

        Returns:
            The metric value

        Raises:
            KeyError: If metric does not exist
        """
        with self._metrics_lock:
            return self._metrics[name]

    def query_events(self, event_type: Optional[str] = None, start_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Query events with optional filtering.

        Uses indexed lookups for efficient querying.

        Args:
            event_type: Optional event type filter
            start_time: Optional start time filter

        Returns:
            List of matching events
        """
        with self._history_lock:
            # Start with all indices
            matching_indices = set(range(len(self._history)))

            # Apply type filter if specified
            if event_type is not None:
                type_indices = set(self._event_index.get(event_type, []))
                matching_indices &= type_indices

            # Apply time filter if specified
            if start_time is not None:
                # Binary search for start time
                time_pos = self._binary_search_time(start_time)
                time_indices = {idx for _, idx in self._time_index[time_pos:]}
                matching_indices &= time_indices

            # Return events in chronological order
            return [self._history[i] for i in sorted(matching_indices)]

    def _binary_search_time(self, target_time: float) -> int:
        """Binary search for the first index >= target_time.

        Args:
            target_time: Target timestamp

        Returns:
            Index of first entry >= target_time
        """
        left, right = 0, len(self._time_index)
        while left < right:
            mid = (left + right) // 2
            if self._time_index[mid][0] < target_time:
                left = mid + 1
            else:
                right = mid
        return left

    def get_metrics_snapshot(self) -> Dict[str, int]:
        """Get a snapshot of all current metrics.

        Returns:
            Dictionary of metric name to value
        """
        with self._metrics_lock:
            return self._metrics.copy()

    def clear_history(self, before_time: Optional[float] = None) -> None:
        """Clear event history.

        Args:
            before_time: Optional timestamp to clear events before
        """
        with self._history_lock:
            if before_time is None:
                self._history.clear()
                self._event_index.clear()
                self._time_index.clear()
                self._event_count = 0
            else:
                # Find cutoff index
                cutoff = self._binary_search_time(before_time)
                remove_indices = {idx for _, idx in self._time_index[:cutoff]}

                # Update history
                new_history = [e for i, e in enumerate(self._history) if i not in remove_indices]
                self._history = new_history

                # Update indices
                self._event_index = {
                    t: [i for i in indices if i not in remove_indices]
                    for t, indices in self._event_index.items()
                }
                self._time_index = self._time_index[cutoff:]
                self._event_count = len(self._history)
