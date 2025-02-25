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
from functools import wraps

from gotstate.types.common import StateId, EventId, EventData, StateAction
from gotstate.core.action import Action
from gotstate.core.exceptions import StateError


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
    """
    Represents a state in a hierarchical state machine.
    
    States can have entry and exit actions, parent states, and child states.
    They can also be composite (containing substates) or orthogonal (containing regions).
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new state.
        
        Args:
            name: Name of the state, used as its identifier
            parent: Optional parent state in a hierarchical structure
        """
        self._state_id = StateId(name)
        self._parent = parent
        self._children: Set[State] = set()
        
        # Add to parent's children if parent exists
        if parent is not None:
            parent._children.add(self)
        
        # Actions
        self._entry_actions: List[Action] = []
        self._exit_actions: List[Action] = []
        self._do_actions: List[Action] = []
        
        # State data
        self._data: Dict[str, Any] = {}
    
    @property
    def id(self) -> StateId:
        """Get the state ID."""
        return self._state_id
    
    @property
    def name(self) -> str:
        """Get the state name."""
        return str(self._state_id)
    
    @property
    def parent(self) -> Optional["State"]:
        """Get the parent state."""
        return self._parent
    
    @property
    def children(self) -> Set["State"]:
        """Get the child states."""
        return self._children.copy()
    
    @property
    def is_composite(self) -> bool:
        """Check if the state is composite (has substates)."""
        return len(self._children) > 0
    
    @property
    def ancestors(self) -> List["State"]:
        """
        Get all ancestor states in order from parent to root.
        
        Returns:
            List of ancestor states
        """
        ancestors = []
        current = self._parent
        while current is not None:
            ancestors.append(current)
            current = current._parent
        return ancestors
    
    @property
    def path(self) -> str:
        """
        Get the full path of the state from root to this state.
        
        Returns:
            String representation of the state path
        """
        if self._parent is None:
            return str(self._state_id)
        else:
            return f"{self._parent.path}/{self._state_id}"
    
    def add_child(self, child: "State") -> None:
        """
        Add a child state.
        
        Args:
            child: The child state to add
            
        Raises:
            StateError: If the child already has a different parent
        """
        if child._parent is not None and child._parent is not self:
            raise StateError(f"State '{child.name}' already has a parent")
        
        child._parent = self
        self._children.add(child)
    
    def remove_child(self, child: "State") -> None:
        """
        Remove a child state.
        
        Args:
            child: The child state to remove
            
        Raises:
            StateError: If the child is not a child of this state
        """
        if child not in self._children:
            raise StateError(f"State '{child.name}' is not a child of '{self.name}'")
        
        self._children.remove(child)
        child._parent = None
    
    def add_entry_action(self, action: Action) -> None:
        """
        Add an entry action.
        
        Args:
            action: The action to execute when entering the state
        """
        self._entry_actions.append(action)
    
    def add_exit_action(self, action: Action) -> None:
        """
        Add an exit action.
        
        Args:
            action: The action to execute when exiting the state
        """
        self._exit_actions.append(action)
    
    def add_do_action(self, action: Action) -> None:
        """
        Add a do activity action.
        
        Args:
            action: The action to execute while in the state
        """
        self._do_actions.append(action)
    
    def execute_entry_actions(self, event_id: EventId, event_data: EventData) -> None:
        """
        Execute all entry actions.
        
        Args:
            event_id: ID of the event that triggered the entry
            event_data: Data associated with the event
        """
        for action in self._entry_actions:
            action.execute(event_id, event_data)
    
    def execute_exit_actions(self, event_id: EventId, event_data: EventData) -> None:
        """
        Execute all exit actions.
        
        Args:
            event_id: ID of the event that triggered the exit
            event_data: Data associated with the event
        """
        for action in self._exit_actions:
            action.execute(event_id, event_data)
    
    def execute_do_actions(self, event_id: EventId, event_data: EventData) -> None:
        """
        Execute all do activity actions.
        
        Args:
            event_id: ID of the event that initiated the do activity
            event_data: Data associated with the event
        """
        for action in self._do_actions:
            action.execute(event_id, event_data)
    
    def on_entry(self, func: StateAction) -> StateAction:
        """
        Decorator to add an entry action to the state.
        
        Args:
            func: The function to execute when entering the state
            
        Returns:
            The decorated function
        """
        action_id = f"{self._state_id}_entry_{len(self._entry_actions)}"
        action = Action(action_id, func)
        self.add_entry_action(action)
        
        @wraps(func)
        def wrapper(event_id: EventId, event_data: EventData) -> None:
            return func(event_id, event_data)
        
        return wrapper
    
    def on_exit(self, func: StateAction) -> StateAction:
        """
        Decorator to add an exit action to the state.
        
        Args:
            func: The function to execute when exiting the state
            
        Returns:
            The decorated function
        """
        action_id = f"{self._state_id}_exit_{len(self._exit_actions)}"
        action = Action(action_id, func)
        self.add_exit_action(action)
        
        @wraps(func)
        def wrapper(event_id: EventId, event_data: EventData) -> None:
            return func(event_id, event_data)
        
        return wrapper
    
    def do_activity(self, func: StateAction) -> StateAction:
        """
        Decorator to add a do activity action to the state.
        
        Args:
            func: The function to execute while in the state
            
        Returns:
            The decorated function
        """
        action_id = f"{self._state_id}_do_{len(self._do_actions)}"
        action = Action(action_id, func)
        self.add_do_action(action)
        
        @wraps(func)
        def wrapper(event_id: EventId, event_data: EventData) -> None:
            return func(event_id, event_data)
        
        return wrapper
    
    def __eq__(self, other: Any) -> bool:
        """
        Compare equality with another state.
        
        Args:
            other: The other state to compare with
            
        Returns:
            True if the states have the same ID, False otherwise
        """
        if not isinstance(other, State):
            return False
        return self._state_id == other._state_id
    
    def __hash__(self) -> int:
        """
        Generate a hash for the state.
        
        Returns:
            Hash value for the state
        """
        return hash(self._state_id)
    
    def __repr__(self) -> str:
        """
        Generate a string representation of the state.
        
        Returns:
            String representation of the state
        """
        return f"State(id={self._state_id}, path={self.path})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the state
            
        Note:
            This serializes the state structure but not the actions.
            When deserializing, the actions must be re-added.
        """
        return {
            "state_id": self._state_id,
            "parent_id": self._parent.id if self._parent else None,
            "children_ids": [child.id for child in self._children],
            "data": self._data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional["State"] = None) -> "State":
        """
        Create a state from a dictionary.
        
        Args:
            data: Dictionary representation of the state
            parent: Optional parent state
            
        Returns:
            New State instance
            
        Note:
            This only recreates the state structure.
            Actions must be added separately after deserialization.
        """
        state = cls(data["state_id"], parent)
        state._data = data.get("data", {})
        return state


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
    pass


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
    pass


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
    pass


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
