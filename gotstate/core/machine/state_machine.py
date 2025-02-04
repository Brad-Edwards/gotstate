from threading import Lock
from typing import Any, Dict, List, Set

from gotstate.core.event import Event
from gotstate.core.machine.machine_status import MachineStatus
from gotstate.core.region import Region
from gotstate.core.state import State
from gotstate.core.transition import Transition


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
        raise NotImplementedError("StateMachine is an abstract base class")

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
            RuntimeError: If termination fails or machine is not in ACTIVE status
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
        raise NotImplementedError("Must be implemented by subclasses")

    def _validate_configuration(self) -> None:
        """Validate machine configuration.

        Called during machine initialization.
        Must be implemented by subclasses.

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If validation fails
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _track_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Track an event with associated data.

        This method provides the default implementation which does nothing.
        Subclasses can override this to implement custom event tracking.

        Args:
            event_name: The name of the event to track
            data: Associated event data
        """
        pass  # Default implementation does nothing

    def _handle_component_error(self, component_type: str, component_id: str, error: Exception) -> None:
        """Handle a component error during operations.

        Template method for handling component errors. The default implementation
        tracks the error via _track_event. Subclasses can override this to
        implement custom error handling.

        Args:
            component_type: Type of component that failed
            component_id: ID of the component that failed
            error: The error that occurred
        """
        self._track_event(
            "component_stop_failure",
            {"component_type": component_type, "component_id": component_id, "error": str(error)},
        )

    def _stop_components(self) -> List[Exception]:
        """Stop all machine components.

        Attempts to stop all components in a best-effort manner,
        collecting any errors that occur during the process.

        Must be implemented by subclasses.

        Returns:
            List of exceptions encountered during stopping
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _start_components(self) -> None:
        """Start all machine components.

        Called during machine activation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _start_event_processing(self) -> None:
        """Start event processing.

        Called during machine activation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _stop_event_processing(self) -> None:
        """Stop event processing.

        Called during machine termination.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _process_single_event(self, event: Event) -> None:
        """Process a single event through the state machine.

        Args:
            event: The event to process
        """
        # Event processing implementation
        pass

    def _cleanup_resources(self) -> None:
        """Clean up managed resources.

        Called during machine termination.
        Must be implemented by subclasses.

        Raises:
            RuntimeError: If cleanup fails
        """
        raise NotImplementedError("Must be implemented by subclasses")

    def _cleanup_components(self) -> None:
        """Clean up machine components.

        Called during initialization failure.
        Must be implemented by subclasses.
        """
        with self._collection_lock:
            self._states.clear()
            self._regions.clear()
            self._transitions.clear()
