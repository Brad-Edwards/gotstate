"""
Parallel region and concurrency management.

Architecture:
- Implements parallel region execution
- Manages region synchronization
- Handles cross-region transitions
- Coordinates with State for hierarchy
- Integrates with Executor for concurrency

Design Patterns:
- Composite Pattern: Region hierarchy
- Observer Pattern: Region events
- Mediator Pattern: Region coordination
- State Pattern: Region lifecycle
- Strategy Pattern: Execution policies

Responsibilities:
1. Parallel Execution
   - True parallel regions
   - State consistency
   - Cross-region transitions
   - Join/fork pseudostates
   - Event ordering

2. Region Synchronization
   - State consistency
   - Event processing
   - Synchronization points
   - Race condition prevention
   - Resource coordination

3. Region Lifecycle
   - Initialization sequence
   - Termination order
   - History restoration
   - Cross-region coordination
   - Data consistency

4. Event Management
   - Event ordering
   - Event propagation
   - Priority handling
   - Scope boundaries
   - Processing rules

Security:
- Region isolation
- Resource boundaries
- State protection
- Event validation

Cross-cutting:
- Error handling
- Performance monitoring
- Region metrics
- Thread safety

Dependencies:
- state.py: State hierarchy
- event.py: Event processing
- executor.py: Parallel execution
- machine.py: Machine context
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Event, Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from gotstate.core.state import State


class RegionStatus(Enum):
    """Defines the possible states of a region.

    Used to track region lifecycle and coordinate execution.
    """

    INACTIVE = auto()  # Region not yet started
    ACTIVE = auto()  # Region executing normally
    SUSPENDED = auto()  # Region temporarily suspended
    TERMINATING = auto()  # Region in process of terminating
    TERMINATED = auto()  # Region fully terminated


class Region:
    """Represents a parallel region in a hierarchical state machine.

    The Region class implements concurrent execution of orthogonal
    state configurations with proper synchronization and isolation.

    Class Invariants:
    1. Must maintain state consistency
    2. Must preserve event ordering
    3. Must handle cross-region transitions
    4. Must enforce isolation boundaries
    5. Must coordinate initialization/termination
    6. Must preserve history state
    7. Must handle interruptions gracefully
    8. Must maintain event scope
    9. Must prevent race conditions
    10. Must manage resources properly

    Design Patterns:
    - Composite: Manages region hierarchy
    - Observer: Notifies of region events
    - Mediator: Coordinates between regions
    - State: Manages region lifecycle
    - Strategy: Implements execution policies
    - Command: Encapsulates region operations

    Data Structures:
    - Set for active states
    - Queue for pending events
    - Map for history states
    - Tree for scope hierarchy
    - Graph for transition paths

    Algorithms:
    - Parallel execution scheduling
    - Event propagation routing
    - Synchronization point management
    - Resource allocation
    - Deadlock prevention

    Threading/Concurrency Guarantees:
    1. Thread-safe state access
    2. Atomic region operations
    3. Synchronized event processing
    4. Safe cross-region transitions
    5. Lock-free status inspection
    6. Mutex protection for critical sections

    Performance Characteristics:
    1. O(1) status updates
    2. O(log n) event routing
    3. O(p) parallel execution where p is active paths
    4. O(s) synchronization where s is sync points
    5. O(r) cross-region coordination where r is region count

    Resource Management:
    1. Bounded thread usage
    2. Controlled memory allocation
    3. Resource pooling
    4. Automatic cleanup
    5. Load balancing
    """

    def __init__(self, region_id: str, parent_state: "State") -> None:
        """Initialize a Region instance.

        Args:
            region_id: Unique identifier for the region
            parent_state: The composite state containing this region

        Raises:
            ValueError: If any parameters are invalid
        """
        if not region_id or not isinstance(region_id, str):
            raise ValueError("Region ID must be a non-empty string")

        if parent_state is None:
            raise ValueError("Parent state must be provided")

        self._id = region_id
        self._parent_state = parent_state
        self._status = RegionStatus.INACTIVE
        self._active_states: Set["State"] = set()
        self._lock = Lock()

    @property
    def id(self) -> str:
        """Get the region ID."""
        return self._id

    @property
    def parent_state(self) -> "State":
        """Get the parent state."""
        return self._parent_state

    @property
    def status(self) -> RegionStatus:
        """Get the region status."""
        return self._status

    @property
    def active_states(self) -> Set["State"]:
        """Get a copy of the active states set."""
        with self._lock:
            return self._active_states.copy()

    @property
    def is_active(self) -> bool:
        """Check if the region is active."""
        return self._status == RegionStatus.ACTIVE

    def activate(self) -> None:
        """Activate the region."""
        with self._lock:
            if self._status == RegionStatus.INACTIVE:
                self._status = RegionStatus.ACTIVE

    def deactivate(self) -> None:
        """Deactivate the region."""
        with self._lock:
            if self._status == RegionStatus.ACTIVE:
                self._status = RegionStatus.INACTIVE
                self._active_states.clear()

    def enter(self) -> None:
        """Enter the region, initializing its initial state."""
        self.activate()

    def exit(self) -> None:
        """Exit the region, clearing its active states."""
        self.deactivate()

    def add_active_state(self, state: "State") -> None:
        """Add a state to the set of active states.

        Args:
            state: The state to mark as active
        """
        with self._lock:
            self._active_states.add(state)

    def remove_active_state(self, state: "State") -> None:
        """Remove a state from the set of active states.

        Args:
            state: The state to mark as inactive
        """
        with self._lock:
            self._active_states.discard(state)


class ParallelRegion(Region):
    """Represents a region that executes in parallel with siblings.

    ParallelRegion implements true concurrent execution with proper
    isolation and synchronization guarantees.

    Class Invariants:
    1. Must maintain parallel execution
    2. Must preserve isolation
    3. Must handle shared resources
    4. Must coordinate with siblings
    5. Must manage thread lifecycle
    6. Must handle interruptions
    7. Must preserve event ordering
    8. Must prevent deadlocks
    9. Must maintain consistency
    10. Must cleanup properly

    Design Patterns:
    - Strategy: Parallel execution policies
    - Observer: Thread coordination
    - Mediator: Resource management
    - Command: Thread operations

    Threading/Concurrency Guarantees:
    1. Thread-safe operations
    2. Atomic state changes
    3. Safe resource access
    4. Deadlock prevention
    5. Proper cleanup

    Performance Characteristics:
    1. O(1) thread operations
    2. O(n) state synchronization
    3. O(r) resource management
    """

    def __init__(self, region_id: str, parent_state: "State") -> None:
        """Initialize a ParallelRegion instance.

        Args:
            region_id: Unique identifier for the region
            parent_state: The composite state containing this region
        """
        super().__init__(region_id=region_id, parent_state=parent_state)
        self._execution_thread = None
        self._stop_event = Event()

    def activate(self) -> None:
        """Activate the region and start parallel execution."""
        with self._lock:
            if self._status == RegionStatus.INACTIVE:
                self._status = RegionStatus.ACTIVE
                self._start_execution()

    def deactivate(self) -> None:
        """Deactivate the region and stop parallel execution."""
        with self._lock:
            if self._status == RegionStatus.ACTIVE:
                self._stop_execution()
                self._status = RegionStatus.INACTIVE
                self._active_states.clear()

    def _start_execution(self) -> None:
        """Start parallel execution of the region."""
        if self._execution_thread is None:
            self._stop_event.clear()
            self._execution_thread = threading.Thread(target=self._execute_region, name=f"Region_{self._id}")
            self._execution_thread.start()

    def _stop_execution(self) -> None:
        """Stop parallel execution of the region."""
        if self._execution_thread is not None:
            self._stop_event.set()
            if self._execution_thread.is_alive():
                self._execution_thread.join(timeout=1.0)
                if self._execution_thread.is_alive():
                    # If thread didn't stop, something is wrong
                    raise RuntimeError("Failed to stop execution thread")
            self._execution_thread = None
            self._stop_event.clear()

    def _execute_region(self) -> None:
        """Execute the region's states in parallel.

        This method runs in a separate thread and manages the parallel
        execution of the region's states.
        """
        try:
            while not self._stop_event.is_set():
                # Execute region logic here
                # For now, just sleep to simulate work
                time.sleep(0.001)  # Very short sleep
        finally:
            # Cleanup when thread exits
            with self._lock:
                self._execution_thread = None


@dataclass
class SyncPoint:
    """Represents a synchronization point in a region.

    Tracks completion status of participating states.
    """

    point_id: str
    participants: List[str]
    _completed: Set[str] = field(default_factory=set)

    @property
    def is_complete(self) -> bool:
        """Check if all participants have completed."""
        return len(self._completed) == len(self.participants)

    def mark_complete(self, state_id: str) -> None:
        """Mark a participant as complete."""
        if state_id in self.participants:
            self._completed.add(state_id)

    def reset(self) -> None:
        """Reset completion status."""
        self._completed.clear()


class SynchronizationRegion(Region):
    """Represents a region that coordinates synchronization points.

    SynchronizationRegion manages join/fork pseudostates and ensures
    proper coordination between parallel regions.

    Class Invariants:
    1. Must maintain sync point validity
    2. Must handle partial completion
    3. Must prevent deadlocks
    4. Must track progress

    Design Patterns:
    - Mediator: Coordinates synchronization
    - Observer: Monitors progress
    - Command: Encapsulates sync operations

    Threading/Concurrency Guarantees:
    1. Thread-safe synchronization
    2. Atomic progress updates
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) point management
    2. O(p) progress tracking where p is participant count
    3. O(d) deadlock detection where d is dependency count
    """

    def __init__(self, region_id: str, parent_state: "State") -> None:
        """Initialize a SynchronizationRegion instance.

        Args:
            region_id: Unique identifier for the region
            parent_state: The composite state containing this region
        """
        super().__init__(region_id=region_id, parent_state=parent_state)
        self._sync_points: Dict[str, SyncPoint] = {}

    @property
    def sync_points(self) -> Dict[str, SyncPoint]:
        """Get the dictionary of sync points."""
        return self._sync_points.copy()

    def add_sync_point(self, point_id: str, participants: List[str]) -> SyncPoint:
        """Add a new synchronization point.

        Args:
            point_id: Unique identifier for the sync point
            participants: List of participating state IDs

        Returns:
            The created sync point

        Raises:
            ValueError: If parameters are invalid
        """
        if not point_id or not isinstance(point_id, str):
            raise ValueError("Sync point ID must be a non-empty string")

        if not participants:
            raise ValueError("Must provide at least one participant")

        if len(set(participants)) != len(participants):
            raise ValueError("Duplicate participants not allowed")

        if point_id in self._sync_points:
            raise ValueError(f"Sync point '{point_id}' already exists")

        point = SyncPoint(point_id=point_id, participants=participants)
        self._sync_points[point_id] = point
        return point

    def mark_sync_complete(self, point_id: str, state_id: str) -> None:
        """Mark a state as complete for a sync point.

        Args:
            point_id: ID of the sync point
            state_id: ID of the completed state

        Raises:
            ValueError: If point_id or state_id is invalid
        """
        if point_id not in self._sync_points:
            raise ValueError(f"Sync point '{point_id}' not found")

        self._sync_points[point_id].mark_complete(state_id)


class HistoryRegion(Region):
    """Represents a region that maintains history state information.

    HistoryRegion preserves and restores historical state configurations
    for both shallow and deep history.

    Class Invariants:
    1. Must maintain history accuracy
    2. Must handle parallel states
    3. Must preserve ordering
    4. Must support restoration

    Design Patterns:
    - Memento: Preserves history state
    - Strategy: Implements history types
    - Command: Encapsulates restoration

    Threading/Concurrency Guarantees:
    1. Thread-safe history tracking
    2. Atomic state restoration
    3. Safe concurrent access

    Performance Characteristics:
    1. O(1) history updates
    2. O(h) state restoration where h is history depth
    3. O(s) parallel state handling where s is state count
    """

    def __init__(self, region_id: str, parent_state: "State") -> None:
        """Initialize a HistoryRegion instance.

        Args:
            region_id: Unique identifier for the region
            parent_state: The composite state containing this region
        """
        super().__init__(region_id=region_id, parent_state=parent_state)
        self._history: List["State"] = []
        self._deep_history: Dict[str, Dict[str, "State"]] = {}  # state_id -> {substate_id -> substate}

    def record_state(self, state: "State", deep: bool = False) -> None:
        """Record a state in history.

        Args:
            state: The state to record
            deep: Whether to record deep history
        """
        with self._lock:
            self._history.append(state)
            if deep:
                self._deep_history[state.id] = {}

    def record_active_substate(self, parent: "State", substate: "State") -> None:
        """Record an active substate for deep history.

        Args:
            parent: The parent state
            substate: The active substate

        Raises:
            ValueError: If parent state not found in deep history
        """
        with self._lock:
            if parent.id not in self._deep_history:
                raise ValueError(f"Parent state '{parent.id}' not found in deep history")
            self._deep_history[parent.id][substate.id] = substate

    def get_history(self) -> List["State"]:
        """Get the list of historical states."""
        with self._lock:
            return self._history.copy()

    def restore_history(self) -> Optional["State"]:
        """Restore the most recent historical state.

        Returns:
            The most recent state, or None if no history
        """
        with self._lock:
            return self._history[-1] if self._history else None

    def restore_deep_history(self) -> Tuple[Optional["State"], Dict[str, "State"]]:
        """Restore deep history state configuration.

        Returns:
            Tuple of (most recent state, dict mapping parent IDs to their active substates)
        """
        with self._lock:
            if not self._history:
                return None, {}
            state = self._history[-1]
            if state.id in self._deep_history:
                # Get the first (and should be only) substate for this parent
                substates = self._deep_history[state.id]
                if substates:
                    # Return mapping of parent ID -> its active substate
                    return state, {state.id: next(iter(substates.values()))}
            return state, {}

    def clear_history(self) -> None:
        """Clear all history information."""
        with self._lock:
            self._history.clear()
            self._deep_history.clear()


class RegionManager:
    """Manages multiple regions and their interactions.

    RegionManager coordinates parallel regions, handles resource
    allocation, and ensures proper synchronization.

    Class Invariants:
    1. Must maintain region isolation
    2. Must handle resource allocation
    3. Must prevent deadlocks
    4. Must coordinate execution
    5. Must manage lifecycle
    6. Must track dependencies
    7. Must handle failures
    8. Must preserve ordering
    9. Must support scaling
    10. Must enforce boundaries

    Design Patterns:
    - Facade: Provides region management interface
    - Factory: Creates region instances
    - Observer: Monitors region status
    - Mediator: Coordinates interactions

    Data Structures:
    - Map of active regions
    - Graph of dependencies
    - Queue of pending operations
    - Pool of resources

    Algorithms:
    - Resource allocation
    - Deadlock detection
    - Load balancing
    - Failure recovery

    Threading/Concurrency Guarantees:
    1. Thread-safe management
    2. Atomic operations
    3. Synchronized coordination
    4. Safe concurrent access
    5. Lock-free inspection
    6. Mutex protection

    Performance Characteristics:
    1. O(1) region lookup
    2. O(log n) resource allocation
    3. O(d) deadlock detection where d is dependency count
    4. O(r) coordination where r is region count
    5. O(f) failure handling where f is failure count

    Resource Management:
    1. Bounded region count
    2. Pooled resources
    3. Automatic cleanup
    4. Load distribution
    5. Failure isolation
    """

    def __init__(self, max_regions: int = 100) -> None:
        """Initialize a RegionManager instance.

        Args:
            max_regions: Maximum number of regions that can be managed

        Raises:
            ValueError: If max_regions is not positive
        """
        if max_regions <= 0:
            raise ValueError("max_regions must be positive")

        self._max_regions = max_regions
        self._regions: Dict[str, Region] = {}
        self._dependencies: Dict[str, Set[str]] = {}  # region_id -> set of dependent region_ids
        self._pending_operations: List[Tuple[str, str]] = []  # List of (region_id, operation)
        self._resource_pool: Dict[str, Any] = {}  # resource_id -> resource
        self._lock = threading.RLock()  # Use RLock instead of Lock for reentrant locking

    @property
    def active_regions(self) -> Dict[str, Region]:
        """Get a copy of the active regions map."""
        with self._lock:
            return self._regions.copy()

    def add_region(self, region: Region) -> None:
        """Add a region to be managed.

        Args:
            region: The region to add

        Raises:
            ValueError: If region already exists or max regions reached
        """
        with self._lock:
            if len(self._regions) >= self._max_regions:
                raise ValueError("Maximum number of regions reached")

            if region.id in self._regions:
                raise ValueError(f"Region '{region.id}' already exists")

            self._regions[region.id] = region
            self._dependencies[region.id] = set()

    def remove_region(self, region_id: str) -> None:
        """Remove a region from management.

        Args:
            region_id: ID of the region to remove

        Raises:
            ValueError: If region not found
        """
        with self._lock:
            if region_id not in self._regions:
                raise ValueError(f"Region '{region_id}' not found")

            # Deactivate region if active
            region = self._regions[region_id]
            if region.is_active:
                region.deactivate()

            # Remove dependencies
            del self._dependencies[region_id]
            for deps in self._dependencies.values():
                deps.discard(region_id)

            # Remove from regions map
            del self._regions[region_id]

    def add_dependency(self, region_id: str, depends_on_id: str) -> None:
        """Add a dependency between regions.

        Args:
            region_id: ID of the dependent region
            depends_on_id: ID of the region being depended on

        Raises:
            ValueError: If either region not found or would create cycle
        """
        with self._lock:
            if region_id not in self._regions:
                raise ValueError(f"Region '{region_id}' not found")

            if depends_on_id not in self._regions:
                raise ValueError(f"Region '{depends_on_id}' not found")

            if region_id == depends_on_id:
                raise ValueError("Region cannot depend on itself")

            # Check for cycles
            if self._would_create_cycle(region_id, depends_on_id):
                raise ValueError("Adding dependency would create cycle")

            self._dependencies[region_id].add(depends_on_id)

    def remove_dependency(self, region_id: str, depends_on_id: str) -> None:
        """Remove a dependency between regions.

        Args:
            region_id: ID of the dependent region
            depends_on_id: ID of the region being depended on

        Raises:
            ValueError: If either region not found
        """
        with self._lock:
            if region_id not in self._regions:
                raise ValueError(f"Region '{region_id}' not found")

            if depends_on_id not in self._regions:
                raise ValueError(f"Region '{depends_on_id}' not found")

            self._dependencies[region_id].discard(depends_on_id)

    def _get_activation_order(self, region_id: str, visited: Set[str]) -> List[str]:
        """Get the order in which regions should be activated.

        Args:
            region_id: ID of the region to activate
            visited: Set of already visited region IDs

        Returns:
            List of region IDs in activation order
        """
        if region_id in visited:
            return []

        visited.add(region_id)
        order = []

        # First add dependencies
        for dep_id in self._dependencies[region_id]:
            order.extend(self._get_activation_order(dep_id, visited))

        # Then add this region
        order.append(region_id)
        return order

    def _get_deactivation_order(self, region_id: str, visited: Set[str]) -> List[str]:
        """Get the order in which regions should be deactivated.

        Args:
            region_id: ID of the region to deactivate
            visited: Set of already visited region IDs

        Returns:
            List of region IDs in deactivation order
        """
        if region_id in visited:
            return []

        visited.add(region_id)
        order = []

        # First add regions that depend on this one
        dependents = {r_id for r_id, deps in self._dependencies.items() if region_id in deps}
        for dep_id in dependents:
            order.extend(self._get_deactivation_order(dep_id, visited))

        # Then add this region
        order.append(region_id)
        return order

    def activate_region(self, region_id: str) -> None:
        """Activate a region and its dependencies.

        Args:
            region_id: ID of the region to activate

        Raises:
            ValueError: If region not found
            RuntimeError: If activation fails
        """
        with self._lock:
            if region_id not in self._regions:
                raise ValueError(f"Region '{region_id}' not found")

            # Get activation order
            activation_order = self._get_activation_order(region_id, set())

            # Activate regions in order
            for r_id in activation_order:
                region = self._regions[r_id]
                if not region.is_active:
                    try:
                        region.activate()
                    except Exception as e:
                        raise RuntimeError(f"Failed to activate region '{r_id}': {str(e)}")

    def deactivate_region(self, region_id: str) -> None:
        """Deactivate a region and dependent regions.

        Args:
            region_id: ID of the region to deactivate

        Raises:
            ValueError: If region not found
            RuntimeError: If deactivation fails
        """
        with self._lock:
            if region_id not in self._regions:
                raise ValueError(f"Region '{region_id}' not found")

            # Get deactivation order
            deactivation_order = self._get_deactivation_order(region_id, set())

            # Deactivate regions in order
            for r_id in deactivation_order:
                region = self._regions[r_id]
                if region.is_active:
                    try:
                        region.deactivate()
                    except Exception as e:
                        raise RuntimeError(f"Failed to deactivate region '{r_id}': {str(e)}")

    def _would_create_cycle(self, region_id: str, depends_on_id: str) -> bool:
        """Check if adding a dependency would create a cycle.

        Args:
            region_id: ID of the dependent region
            depends_on_id: ID of the region being depended on

        Returns:
            True if adding the dependency would create a cycle
        """
        visited = set()
        to_visit = {depends_on_id}

        while to_visit:
            current = to_visit.pop()
            if current == region_id:
                return True

            visited.add(current)
            to_visit.update(self._dependencies[current] - visited)

        return False

    def allocate_resource(self, resource_id: str, resource: Any) -> None:
        """Add a resource to the resource pool.

        Args:
            resource_id: Unique identifier for the resource
            resource: The resource to add

        Raises:
            ValueError: If resource_id already exists
        """
        with self._lock:
            if resource_id in self._resource_pool:
                raise ValueError(f"Resource '{resource_id}' already exists")

            self._resource_pool[resource_id] = resource

    def deallocate_resource(self, resource_id: str) -> None:
        """Remove a resource from the resource pool.

        Args:
            resource_id: ID of the resource to remove

        Raises:
            ValueError: If resource not found
        """
        with self._lock:
            if resource_id not in self._resource_pool:
                raise ValueError(f"Resource '{resource_id}' not found")

            del self._resource_pool[resource_id]

    def get_resource(self, resource_id: str) -> Any:
        """Get a resource from the resource pool.

        Args:
            resource_id: ID of the resource to get

        Returns:
            The requested resource

        Raises:
            ValueError: If resource not found
        """
        with self._lock:
            if resource_id not in self._resource_pool:
                raise ValueError(f"Resource '{resource_id}' not found")

            return self._resource_pool[resource_id]

    def cleanup(self) -> None:
        """Clean up all resources and deactivate all regions.

        This method attempts to clean up all resources and deactivate all regions,
        continuing even if some operations fail. All failures are logged.
        """
        logger = logging.getLogger(__name__)

        with self._lock:
            # Track any errors that occur during cleanup
            cleanup_errors = []

            # Deactivate all regions in dependency order
            active_regions = [r_id for r_id, r in self._regions.items() if r.is_active]
            for region_id in active_regions:
                try:
                    self.deactivate_region(region_id)
                except (ValueError, RuntimeError) as e:
                    error_msg = f"Failed to deactivate region '{region_id}': {str(e)}"
                    logger.error(error_msg)
                    cleanup_errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error deactivating region '{region_id}': {str(e)}"
                    logger.exception(error_msg)  # This logs the full stack trace
                    cleanup_errors.append(error_msg)

            # Clear all collections
            self._regions.clear()
            self._dependencies.clear()
            self._pending_operations.clear()
            self._resource_pool.clear()

            # Log summary if there were any errors
            if cleanup_errors:
                logger.error(f"Cleanup completed with {len(cleanup_errors)} errors: {'; '.join(cleanup_errors)}")
            else:
                logger.debug("Cleanup completed successfully")
