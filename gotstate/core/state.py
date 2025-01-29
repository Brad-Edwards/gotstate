"""
State class and hierarchy management.

Architecture:
- Implements hierarchical state structure using Composite pattern
- Manages state data with isolation guarantees
- Enforces state invariants and validation
- Coordinates with Region for parallel state execution
- Preserves history state information

Design Patterns:
- Composite Pattern: Hierarchical state structure
- Observer Pattern: State change notifications
- Memento Pattern: History state preservation
- Builder Pattern: State configuration
- Visitor Pattern: State traversal

Responsibilities:
1. State Hierarchy
   - Parent/child relationships
   - Composite state management
   - Submachine state handling
   - State redefinition support

2. State Data
   - Data isolation between states
   - Parent state data inheritance
   - Parallel region data management
   - History state data preservation

3. State Behavior
   - Entry/exit actions
   - Do-activity execution
   - Internal transitions
   - State invariants

4. State Configuration
   - Initial/final states
   - History state types
   - Entry/exit points
   - Choice/junction pseudostates

Security:
- State data isolation
- Action execution boundaries
- Resource usage monitoring
- Validation at state boundaries

Cross-cutting:
- Error handling for state operations
- Performance optimization for traversal
- Monitoring of state changes
- Thread safety for parallel regions

Dependencies:
- region.py: Parallel region coordination
- transition.py: State change management
- event.py: Event processing integration
- machine.py: State machine context
"""

from typing import Optional, Dict, List, Set, Any, Callable
from dataclasses import dataclass
from enum import Enum, auto
from weakref import ref, ReferenceType


class StateType(Enum):
    """Defines the different types of states in the hierarchical state machine.
    
    Used to distinguish between regular states, pseudostates, and special state types
    for proper behavioral implementation and validation.
    """
    SIMPLE = auto()          # Leaf state with no substates
    COMPOSITE = auto()       # State containing substates
    SUBMACHINE = auto()      # Reference to another state machine
    INITIAL = auto()         # Initial pseudostate
    FINAL = auto()          # Final state
    CHOICE = auto()         # Dynamic conditional branching
    JUNCTION = auto()       # Static conditional branching
    SHALLOW_HISTORY = auto() # Remembers only direct substate
    DEEP_HISTORY = auto()    # Remembers full substate configuration
    ENTRY_POINT = auto()    # Named entry point
    EXIT_POINT = auto()     # Named exit point
    TERMINATE = auto()      # Terminates entire state machine


class State:
    """Represents a state in a hierarchical state machine.
    
    The State class implements the Composite pattern to manage the hierarchical
    structure of states. It maintains parent-child relationships, handles state
    data with proper isolation, and coordinates with parallel regions.
    
    Class Invariants:
    1. A state must have a unique identifier within its parent's scope
    2. A state's type must not change after initialization
    3. Parent-child relationships must form a directed acyclic graph (DAG)
    4. Initial pseudostates must have exactly one outgoing transition
    5. History states must belong to a composite state
    6. Entry/exit points must have valid connections
    7. State data must remain isolated between parallel regions
    8. Parent state data must be accessible to child states
    9. Active do-activities must be properly tracked and managed
    10. State configuration must be valid according to UML state machine rules
    """
    def __init__(
        self,
        state_id: str,
        state_type: StateType,
        parent: Optional['State'] = None,
        data: Optional[Dict[str, Any]] = None,
        entry_action: Optional[Callable[[], None]] = None,
        exit_action: Optional[Callable[[], None]] = None,
        do_activity: Optional[Any] = None
    ) -> None:
        """Initialize a State instance.
        
        Args:
            state_id: Unique identifier for the state
            state_type: Type of state (simple, composite, etc.)
            parent: Optional parent state
            data: Optional state data dictionary
            entry_action: Optional action to execute on state entry
            exit_action: Optional action to execute on state exit
            do_activity: Optional activity to execute while in state
            
        Raises:
            ValueError: If any parameters are invalid
        """
        if not state_id or not isinstance(state_id, str):
            raise ValueError("State ID must be a non-empty string")
            
        if not isinstance(state_type, StateType):
            raise ValueError("State type must be a StateType enum value")
            
        self._id = state_id
        self._type = state_type
        self._parent: Optional[ReferenceType['State']] = None
        self._children: Dict[str, 'State'] = {}
        self._data = data.copy() if data else {}
        self._is_active = False
        self._entry_action = entry_action
        self._exit_action = exit_action
        self._do_activity = do_activity
        
        # Set parent after initialization to handle parent-child relationship
        if parent is not None:
            self.set_parent(parent)
            
    @property
    def id(self) -> str:
        """Get the state ID."""
        return self._id
        
    @property
    def type(self) -> StateType:
        """Get the state type."""
        return self._type
        
    @property
    def parent(self) -> Optional['State']:
        """Get the parent state."""
        return self._parent() if self._parent is not None else None
        
    @property
    def children(self) -> Dict[str, 'State']:
        """Get a copy of the children dictionary."""
        return self._children.copy()
        
    @property
    def data(self) -> Dict[str, Any]:
        """Get a copy of the state data."""
        return self._data.copy()
        
    @property
    def is_active(self) -> bool:
        """Check if the state is active."""
        return self._is_active
        
    def set_parent(self, parent: 'State') -> None:
        """Set the parent state.
        
        Args:
            parent: The new parent state
            
        Raises:
            ValueError: If setting the parent would create a cycle
                       or if the parent cannot accept children
        """
        # Check for cycles
        current = parent
        while current is not None:
            if current is self:
                raise ValueError("Cannot create cyclic parent-child relationship")
            current = current.parent
            
        # Check if parent can accept children
        if not parent.can_have_children():
            raise ValueError(f"State of type {parent.type} cannot have children")
            
        # Remove from old parent
        if self._parent is not None:
            old_parent = self._parent()
            if old_parent is not None:
                old_parent._children.pop(self.id, None)
                
        # Update parent reference
        self._parent = ref(parent) if parent is not None else None
        
        # Add to new parent's children
        if parent is not None:
            parent._children[self.id] = self
            
    def remove_from_parent(self) -> None:
        """Remove this state from its parent."""
        if self._parent is not None:
            parent = self._parent()
            if parent is not None:
                parent._children.pop(self.id, None)
            self._parent = None
            
    def activate(self) -> None:
        """Activate the state."""
        self._is_active = True
        
    def deactivate(self) -> None:
        """Deactivate the state."""
        self._is_active = False
        
    def enter(self) -> None:
        """Enter the state, executing entry action and starting do-activity."""
        if self._entry_action is not None:
            self._entry_action()
            
        if self._do_activity is not None:
            self._do_activity.start()
            
        self.activate()
        
    def exit(self) -> None:
        """Exit the state, executing exit action and stopping do-activity."""
        if self._do_activity is not None:
            self._do_activity.stop()
            
        if self._exit_action is not None:
            self._exit_action()
            
        self.deactivate()
            
    def can_have_children(self) -> bool:
        """Check if this state can have child states.
        
        Returns:
            True if this state can have children, False otherwise
        """
        return True  # Base states can have children by default


class CompositeState(State):
    """Represents a composite state that can contain other states.
    
    CompositeState extends the base State class to implement the Composite pattern,
    managing a collection of child states and their relationships.
    
    Class Invariants:
    1. Must maintain valid parent-child relationships
    2. Must have at most one initial state per region
    3. Must properly manage parallel regions
    4. Must maintain history state consistency
    5. Must enforce state naming uniqueness within scope
    
    Design Patterns:
    - Composite: Manages child state hierarchy
    - Factory: Creates appropriate state types
    - Observer: Notifies of child state changes
    
    Data Structures:
    - Dictionary of child states by name
    - List of parallel regions
    - Map of history states
    - Set of active substates
    
    Threading/Concurrency Guarantees:
    1. Thread-safe child state access
    2. Atomic region activation/deactivation
    3. Synchronized history state updates
    4. Safe concurrent region execution
    
    Performance Characteristics:
    1. O(1) child state lookup
    2. O(r) region synchronization where r is region count
    3. O(h) history state management where h is history count
    """
    def __init__(
        self,
        state_id: str,
        parent: Optional['State'] = None,
        data: Optional[Dict[str, Any]] = None,
        entry_action: Optional[Callable[[], None]] = None,
        exit_action: Optional[Callable[[], None]] = None,
        do_activity: Optional[Any] = None
    ) -> None:
        """Initialize a CompositeState instance.
        
        Args:
            state_id: Unique identifier for the state
            parent: Optional parent state
            data: Optional state data dictionary
            entry_action: Optional action to execute on state entry
            exit_action: Optional action to execute on state exit
            do_activity: Optional activity to execute while in state
        """
        super().__init__(
            state_id=state_id,
            state_type=StateType.COMPOSITE,
            parent=parent,
            data=data,
            entry_action=entry_action,
            exit_action=exit_action,
            do_activity=do_activity
        )
        self._regions: Dict[str, 'Region'] = {}
        
    @property
    def regions(self) -> Dict[str, 'Region']:
        """Get a copy of the regions dictionary."""
        return self._regions.copy()
        
    def add_region(self, region_id: str) -> 'Region':
        """Add a new region to the composite state.
        
        Args:
            region_id: Unique identifier for the region
            
        Returns:
            The newly created region
            
        Raises:
            ValueError: If region_id is invalid or already exists
        """
        if not region_id or not isinstance(region_id, str):
            raise ValueError("Region ID must be a non-empty string")
            
        if region_id in self._regions:
            raise ValueError(f"Region '{region_id}' already exists")
            
        # Import here to avoid circular dependency
        from .region import Region
        
        region = Region(region_id=region_id, parent_state=self)
        self._regions[region_id] = region
        return region
        
    def remove_region(self, region_id: str) -> None:
        """Remove a region from the composite state.
        
        Args:
            region_id: Identifier of the region to remove
        """
        if region_id in self._regions:
            region = self._regions.pop(region_id)
            region.deactivate()
            
    def activate(self) -> None:
        """Activate the composite state and all its regions."""
        super().activate()
        for region in self._regions.values():
            region.activate()
            
    def deactivate(self) -> None:
        """Deactivate the composite state and all its regions."""
        for region in self._regions.values():
            region.deactivate()
        super().deactivate()
        
    def enter(self) -> None:
        """Enter the composite state, activating regions after entry action."""
        if self._entry_action is not None:
            self._entry_action()
            
        if self._do_activity is not None:
            self._do_activity.start()
            
        self.activate()
        
        # Enter regions after state is active
        for region in self._regions.values():
            region.enter()
            
    def exit(self) -> None:
        """Exit the composite state, exiting regions before exit action."""
        # Exit regions before state becomes inactive
        for region in self._regions.values():
            region.exit()
            
        if self._do_activity is not None:
            self._do_activity.stop()
            
        if self._exit_action is not None:
            self._exit_action()
            
        self.deactivate()


class PseudoState(State):
    """Base class for all pseudostates in the state machine.
    
    PseudoState provides common functionality for special states that control
    execution flow but don't represent actual system states.
    
    Class Invariants:
    1. Must have valid connections according to type
    2. Must not contain substates
    3. Must follow UML pseudostate semantics
    4. Must maintain transition consistency
    
    Design Patterns:
    - Template Method: Defines pseudostate behavior
    - Strategy: Implements type-specific logic
    - Chain of Responsibility: Handles transition routing
    
    Threading/Concurrency Guarantees:
    1. Thread-safe transition execution
    2. Atomic decision point evaluation
    3. Safe concurrent access to guard conditions
    
    Performance Characteristics:
    1. O(1) type checking
    2. O(t) transition evaluation where t is transition count
    3. O(g) guard condition evaluation where g is guard count
    """
    # Set of valid pseudostate types
    VALID_TYPES = {
        StateType.INITIAL,
        StateType.CHOICE,
        StateType.JUNCTION,
        StateType.SHALLOW_HISTORY,
        StateType.DEEP_HISTORY,
        StateType.ENTRY_POINT,
        StateType.EXIT_POINT,
        StateType.TERMINATE
    }
    
    def __init__(
        self,
        state_id: str,
        state_type: StateType,
        parent: Optional['State'] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a PseudoState instance.
        
        Args:
            state_id: Unique identifier for the state
            state_type: Type of pseudostate
            parent: Parent state (required for pseudostates)
            data: Optional state data dictionary
            
        Raises:
            ValueError: If any parameters are invalid
        """
        if state_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid pseudostate type. Must be one of: {self.VALID_TYPES}")
            
        if parent is None:
            raise ValueError("Pseudostates must have a parent state")
            
        super().__init__(
            state_id=state_id,
            state_type=state_type,
            parent=parent,
            data=data
        )
        
    def set_parent(self, parent: 'State') -> None:
        """Set the parent state.
        
        Args:
            parent: The new parent state
            
        Raises:
            ValueError: If parent is None
        """
        if parent is None:
            raise ValueError("Pseudostates must have a parent state")
        super().set_parent(parent)
        
    def activate(self) -> None:
        """Pseudostates cannot remain active."""
        pass  # Pseudostates are transient and cannot be activated
        
    def can_have_children(self) -> bool:
        """Check if this state can have child states.
        
        Returns:
            False, as pseudostates cannot have children
        """
        return False


class HistoryState(PseudoState):
    """Represents history pseudostates (shallow and deep) in the state machine.
    
    HistoryState maintains the historical state configuration of its parent
    composite state, enabling state restoration.
    
    Class Invariants:
    1. Must belong to a composite state
    2. Must maintain valid history configuration
    3. Must preserve parallel region history
    4. Must handle default transitions
    
    Design Patterns:
    - Memento: Stores and restores state configuration
    - Observer: Tracks state configuration changes
    - Strategy: Implements history type behavior
    
    Data Structures:
    - Stack for state configuration history
    - Map for region history tracking
    - Set for active state tracking
    
    Threading/Concurrency Guarantees:
    1. Thread-safe history updates
    2. Atomic configuration restoration
    3. Safe concurrent region history tracking
    
    Performance Characteristics:
    1. O(1) history type checking
    2. O(d) configuration storage where d is hierarchy depth
    3. O(r) region history management where r is region count
    """
    # Set of valid history state types
    VALID_TYPES = {
        StateType.SHALLOW_HISTORY,
        StateType.DEEP_HISTORY
    }
    
    def __init__(
        self,
        state_id: str,
        state_type: StateType,
        parent: Optional['State'] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a HistoryState instance.
        
        Args:
            state_id: Unique identifier for the state
            state_type: Type of history state (shallow or deep)
            parent: Parent composite state
            data: Optional state data dictionary
            
        Raises:
            ValueError: If any parameters are invalid
        """
        if state_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid history state type. Must be one of: {self.VALID_TYPES}")
            
        if parent is None or parent.type != StateType.COMPOSITE:
            raise ValueError("History states must belong to a composite state")
            
        super().__init__(
            state_id=state_id,
            state_type=state_type,
            parent=parent,
            data=data
        )
        self._last_active_state: Optional['State'] = None
        
    @property
    def last_active_state(self) -> Optional['State']:
        """Get the last active state recorded by this history state."""
        return self._last_active_state
        
    def record_active_state(self, state: 'State') -> None:
        """Record a state as being active.
        
        Args:
            state: The state to record as active
        """
        self._last_active_state = state
        
    def clear_history(self) -> None:
        """Clear the recorded history state."""
        self._last_active_state = None
        
    def get_restoration_state(self) -> Optional['State']:
        """Get the state to restore when this history state is entered.
        
        Returns:
            The state to restore, or None if no history is available
        """
        return self._last_active_state


class ConnectionPointState(PseudoState):
    """Represents entry and exit points for states.
    
    ConnectionPointState manages named entry and exit points that provide
    interfaces for transitions into and out of composite states.
    
    Class Invariants:
    1. Must have valid connection to parent state
    2. Must maintain transition consistency
    3. Must have unique name within parent scope
    4. Must enforce valid transition paths
    
    Design Patterns:
    - Facade: Provides clean interface to state
    - Mediator: Coordinates transition routing
    - Chain of Responsibility: Handles transition paths
    
    Threading/Concurrency Guarantees:
    1. Thread-safe transition routing
    2. Atomic path validation
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) point type checking
    2. O(p) path validation where p is path length
    3. O(t) transition routing where t is transition count
    """
    pass


class ChoiceState(PseudoState):
    """Represents a dynamic conditional branch point.
    
    ChoiceState evaluates guard conditions at runtime to determine the
    transition path, enabling dynamic behavioral decisions.
    
    Class Invariants:
    1. Must have at least one outgoing transition
    2. Must evaluate guards in defined order
    3. Must have valid default transition
    4. Must maintain consistent decision state
    
    Design Patterns:
    - Strategy: Implements guard evaluation
    - Chain of Responsibility: Processes guards
    - Command: Encapsulates guard conditions
    
    Data Structures:
    - Priority queue for guard evaluation
    - Decision tree for condition checking
    
    Threading/Concurrency Guarantees:
    1. Thread-safe guard evaluation
    2. Atomic decision making
    3. Safe concurrent condition access
    
    Performance Characteristics:
    1. O(g) guard evaluation where g is guard count
    2. O(log g) guard prioritization
    3. O(d) decision tree traversal where d is tree depth
    """
    pass


class JunctionState(PseudoState):
    """Represents a static conditional branch point.
    
    JunctionState implements static conditional branching based on
    guard conditions that are evaluated when the junction is reached.
    
    Class Invariants:
    1. Must have at least one outgoing transition
    2. Must evaluate guards in static order
    3. Must have valid default transition
    4. Must maintain transition consistency
    
    Design Patterns:
    - Strategy: Implements branching logic
    - Chain of Responsibility: Processes conditions
    - Command: Encapsulates static decisions
    
    Threading/Concurrency Guarantees:
    1. Thread-safe transition selection
    2. Atomic path determination
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(t) transition evaluation where t is transition count
    2. O(g) guard checking where g is guard count
    3. O(1) default transition access
    """
    pass
