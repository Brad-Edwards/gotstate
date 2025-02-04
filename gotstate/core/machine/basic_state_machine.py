import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from typing import Any, Callable, Dict, Set

from gotstate.core.event import Event, EventQueue
from gotstate.core.machine.machine_monitor import MachineMonitor
from gotstate.core.machine.machine_status import MachineStatus
from gotstate.core.machine.state_machine import StateMachine
from gotstate.core.region import Region
from gotstate.core.state import State
from gotstate.core.transition import Transition


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
            "max_resources": 100,
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
            RuntimeError: If termination fails or machine is not in ACTIVE status
        """
        with self._status_lock:
            if self._status == MachineStatus.TERMINATED:
                return

            if self._status != MachineStatus.ACTIVE:
                raise RuntimeError("Machine must be in ACTIVE status to terminate")

            prev_status = self._status
            self._status = MachineStatus.TERMINATING

        try:
            # Stop processing
            self._stop_event_processing()

            # Stop components (collecting any errors)
            component_errors = self._stop_components()

            # Always attempt cleanup
            try:
                self._cleanup_resources()
            except Exception as cleanup_error:
                # If we had no component errors, use the cleanup error
                if not component_errors:
                    component_errors = [cleanup_error]

            # If we had any errors, restore status and raise the first one
            if component_errors:
                with self._status_lock:
                    self._status = prev_status
                raise RuntimeError(f"Termination failed: {component_errors[0]}")

            # Set TERMINATED only if everything succeeds
            with self._status_lock:
                self._status = MachineStatus.TERMINATED

        except Exception as e:
            # Handle any other unexpected errors
            with self._status_lock:
                self._status = prev_status
            if isinstance(e, RuntimeError) and "Termination failed:" in str(e):
                raise e  # Don't wrap already wrapped errors
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
                self._track_event(
                    "event_processing", {"event_id": id(event), "event_kind": event.kind.name, "event_data": event.data}
                )
                self._event_queue.enqueue(event)
                self._track_event("event_queued", {"event_id": id(event), "queue_size": len(self._event_queue)})
            except Exception as e:
                self._track_event("event_processing_failure", {"event_id": id(event), "error": str(e)})
                raise

    def _track_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Track a machine event with monitoring.

        Args:
            event_type: Type of event
            details: Event details
        """
        with self._monitor_lock:
            event_data = {"type": event_type, "timestamp": time.time(), "machine_status": self.status.name, **details}
            self._monitor.track_event(event_data)

    def _initialize_components(self) -> None:
        """Initialize all machine components."""
        # Collect all components first to minimize lock time
        with self._collection_lock:
            states = list(self._states.values())
            regions = list(self._regions.values())

        # Batch track initialization start
        events = []
        for state in states:
            events.append(("component_init", {"component_type": "state", "component_id": state.id}))
        for region in regions:
            events.append(("component_init", {"component_type": "region", "component_id": region.id}))

        # Track all events at once
        with self._monitor_lock:
            for event_type, event_data in events:
                self._track_event(event_type, event_data)

        # Initialize states in parallel if we have any
        if states:

            def init_state(state):
                try:
                    state.initialize()
                    return ("success", state.id, None)
                except Exception as e:
                    return ("failure", state.id, e)

            with ThreadPoolExecutor(max_workers=min(32, len(states))) as executor:
                future_to_state = {executor.submit(init_state, state): state for state in states}
                for future in as_completed(future_to_state):
                    status, state_id, error = future.result()
                    if status == "success":
                        self._track_event(
                            "component_init_success", {"component_type": "state", "component_id": state_id}
                        )
                    else:
                        self._track_event(
                            "component_init_failure",
                            {"component_type": "state", "component_id": state_id, "error": str(error)},
                        )
                        raise error

        # Initialize regions (keep sequential as they may have dependencies)
        for region in regions:
            try:
                region.initialize()
                self._track_event("component_init_success", {"component_type": "region", "component_id": region.id})
            except Exception as e:
                self._track_event(
                    "component_init_failure", {"component_type": "region", "component_id": region.id, "error": str(e)}
                )
                raise

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

    def _start_event_processing(self) -> None:
        """Start event processing."""
        with self._processing_lock:
            self._event_queue.start_processing()

    def _stop_event_processing(self) -> None:
        """Stop event processing."""
        with self._processing_lock:
            self._event_queue.stop_processing()

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
            self._track_event("security_policy_added", {"operation": operation})

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
            self._track_event("resource_limit_set", {"resource": resource, "limit": limit})

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
