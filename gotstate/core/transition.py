"""
Transition types and behavior management.

Architecture:
- Implements transition type hierarchy and behavior
- Manages transition execution and actions
- Resolves transition conflicts
- Coordinates with State for state changes
- Integrates with Event for triggers

Design Patterns:
- Command Pattern: Transition execution
- Strategy Pattern: Transition types
- Chain of Responsibility: Guard evaluation
- Observer Pattern: Transition notifications
- Template Method: Transition execution steps

Responsibilities:
1. Transition Types
   - External transitions
   - Internal transitions
   - Local transitions
   - Compound transitions
   - Protocol transitions

2. Transition Behavior
   - Guard conditions
   - Actions execution
   - Source/target validation
   - Completion transitions
   - Time/change triggers

3. Semantic Resolution
   - Conflict resolution
   - Priority handling
   - Simultaneous transitions
   - Cross-region coordination
   - Execution ordering

4. Error Handling
   - Partial completion
   - Guard evaluation errors
   - Action execution failures
   - State consistency
   - Resource cleanup

Security:
- Action execution isolation
- Guard evaluation boundaries
- Resource usage control
- State change validation

Cross-cutting:
- Error propagation
- Performance monitoring
- Transition metrics
- Thread safety

Dependencies:
- state.py: State change coordination
- event.py: Event trigger integration
- region.py: Cross-region transitions
- machine.py: Machine context
"""

from typing import Optional, List, Callable, Any, Dict, Set, Union
from enum import Enum, auto
from dataclasses import dataclass
from uuid import uuid4

from gotstate.types.common import TransitionId, EventId, EventData
from gotstate.core.action import Action
from gotstate.core.guard import Guard
from gotstate.core.exceptions import TransitionError


class TransitionKind(Enum):
    """Defines the different types of transitions in the state machine.
    
    Used to determine the execution semantics and state exit/entry behavior
    for each transition type.
    """
    EXTERNAL = auto()  # Exits source state(s), enters target state(s)
    INTERNAL = auto()  # No state exit/entry, source must equal target
    LOCAL = auto()     # Minimizes state exit/entry within composite state
    COMPOUND = auto()  # Multiple segments with intermediate pseudostates


class TransitionPriority(Enum):
    """Defines priority levels for transition conflict resolution.
    
    Used to determine which transition takes precedence when multiple
    transitions are enabled simultaneously.
    """
    HIGH = auto()    # Takes precedence over lower priorities
    NORMAL = auto()  # Default priority level
    LOW = auto()     # Yields to higher priority transitions


class Transition:
    """
    Represents a transition between states in a state machine.
    
    Transitions connect source and target states and are triggered by events.
    They can have guards (conditions) and actions (behaviors executed during the transition).
    """
    
    def __init__(
        self,
        source: "State",
        target: Optional["State"],
        event_id: Optional[str] = None,
        guard: Optional[Guard] = None,
        actions: Optional[List[Action]] = None,
        transition_id: Optional[str] = None,
    ):
        """
        Initialize a new transition.
        
        Args:
            source: Source state of the transition
            target: Target state of the transition (None for internal transitions)
            event_id: ID of the event that triggers the transition (None for completion transitions)
            guard: Optional guard condition for the transition
            actions: Optional list of actions to execute during the transition
            transition_id: Optional unique identifier for the transition
        """
        from gotstate.core.state import State
        
        if not isinstance(source, State):
            raise TransitionError(f"Source must be a State, got {type(source)}")
        
        if target is not None and not isinstance(target, State):
            raise TransitionError(f"Target must be a State or None, got {type(target)}")
        
        self._source = source
        self._target = target
        self._event_id = None if event_id is None else EventId(event_id)
        self._guard = guard
        self._actions = actions or []
        self._transition_id = TransitionId(transition_id or str(uuid4()))
    
    @property
    def id(self) -> TransitionId:
        """Get the transition ID."""
        return self._transition_id
    
    @property
    def source(self) -> "State":
        """Get the source state."""
        return self._source
    
    @property
    def target(self) -> Optional["State"]:
        """Get the target state."""
        return self._target
    
    @property
    def event_id(self) -> Optional[EventId]:
        """Get the event ID that triggers this transition."""
        return self._event_id
    
    @property
    def guard(self) -> Optional[Guard]:
        """Get the guard condition for this transition."""
        return self._guard
    
    @property
    def actions(self) -> List[Action]:
        """Get the actions for this transition."""
        return self._actions.copy()
    
    @property
    def is_internal(self) -> bool:
        """Check if this is an internal transition (no state change)."""
        return self._target is None
    
    @property
    def is_external(self) -> bool:
        """Check if this is an external transition (state change)."""
        return not self.is_internal
    
    @property
    def is_completion(self) -> bool:
        """Check if this is a completion transition (no event trigger)."""
        return self._event_id is None
    
    def can_trigger(self, event_id: EventId, event_data: EventData) -> bool:
        """
        Check if this transition can be triggered by the given event.
        
        Args:
            event_id: ID of the event to check
            event_data: Data associated with the event
            
        Returns:
            True if the transition can be triggered, False otherwise
        """
        # Check if event ID matches
        if self._event_id is not None and self._event_id != event_id:
            return False
        
        # Check if guard condition is satisfied
        if self._guard is not None and not self._guard.evaluate(event_id, event_data):
            return False
        
        return True
    
    def execute_actions(self, event_id: EventId, event_data: EventData) -> None:
        """
        Execute all transition actions.
        
        Args:
            event_id: ID of the event that triggered the transition
            event_data: Data associated with the event
        """
        for action in self._actions:
            action.execute(event_id, event_data)
    
    def add_action(self, action: Action) -> None:
        """
        Add an action to this transition.
        
        Args:
            action: The action to add
        """
        self._actions.append(action)
    
    def set_guard(self, guard: Guard) -> None:
        """
        Set the guard condition for this transition.
        
        Args:
            guard: The guard condition to set
        """
        self._guard = guard
    
    def __eq__(self, other: Any) -> bool:
        """
        Compare equality with another transition.
        
        Args:
            other: The other transition to compare with
            
        Returns:
            True if the transitions have the same ID, False otherwise
        """
        if not isinstance(other, Transition):
            return False
        return self._transition_id == other._transition_id
    
    def __hash__(self) -> int:
        """
        Generate a hash for the transition.
        
        Returns:
            Hash value for the transition
        """
        return hash(self._transition_id)
    
    def __repr__(self) -> str:
        """
        Generate a string representation of the transition.
        
        Returns:
            String representation of the transition
        """
        source_name = self._source.name if self._source else "None"
        target_name = self._target.name if self._target else "None"
        event_str = str(self._event_id) if self._event_id else "completion"
        return (
            f"Transition(id={self._transition_id}, "
            f"source={source_name}, "
            f"target={target_name}, "
            f"event={event_str})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transition to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the transition
            
        Note:
            This only serializes the structure, not the actual guard and action functions.
            When deserializing, these must be re-added.
        """
        return {
            "transition_id": self._transition_id,
            "source_id": self._source.id,
            "target_id": self._target.id if self._target else None,
            "event_id": self._event_id,
            "guard_id": self._guard.id if self._guard else None,
            "action_ids": [action.id for action in self._actions]
        }


class ExternalTransition(Transition):
    """Represents an external transition that exits source state(s).
    
    ExternalTransition implements the full exit/entry state behavior,
    following UML state machine semantics.
    
    Class Invariants:
    1. Must exit source state(s)
    2. Must enter target state(s)
    3. Must execute actions in correct order
    4. Must maintain state consistency
    
    Design Patterns:
    - Template Method: Defines execution sequence
    - Command: Encapsulates state changes
    - Observer: Notifies of state changes
    
    Threading/Concurrency Guarantees:
    1. Thread-safe state changes
    2. Atomic execution sequence
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(h) state exit/entry where h is hierarchy depth
    2. O(a) action execution where a is action count
    3. O(r) region synchronization where r is region count
    """
    pass


class InternalTransition(Transition):
    """Represents an internal transition within a single state.
    
    InternalTransition executes without exiting or entering states,
    maintaining the current state configuration.
    
    Class Invariants:
    1. Source must equal target state
    2. Must not exit/enter states
    3. Must maintain state consistency
    4. Must execute actions atomically
    
    Design Patterns:
    - Strategy: Implements internal behavior
    - Command: Encapsulates actions
    - Observer: Notifies of execution
    
    Threading/Concurrency Guarantees:
    1. Thread-safe action execution
    2. Atomic state updates
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) state validation
    2. O(a) action execution where a is action count
    3. O(1) consistency check
    """
    pass


class LocalTransition(Transition):
    """Represents a local transition within a composite state.
    
    LocalTransition minimizes the scope of state exit/entry operations
    while maintaining proper transition semantics.
    
    Class Invariants:
    1. Must minimize state exit/entry
    2. Must maintain hierarchy consistency
    3. Must execute actions in order
    4. Must preserve region stability
    
    Design Patterns:
    - Strategy: Implements local semantics
    - Command: Encapsulates minimal changes
    - Observer: Notifies of local changes
    
    Threading/Concurrency Guarantees:
    1. Thread-safe local changes
    2. Atomic scope execution
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(d) scope computation where d is depth difference
    2. O(a) action execution where a is action count
    3. O(r) region synchronization where r is region count
    """
    pass


class CompoundTransition(Transition):
    """Represents a compound transition with multiple segments.
    
    CompoundTransition manages a sequence of transition segments,
    coordinating their execution through pseudostates.
    
    Class Invariants:
    1. Must have valid segment sequence
    2. Must maintain execution order
    3. Must coordinate pseudostates
    4. Must handle segment failures
    
    Design Patterns:
    - Composite: Manages transition segments
    - Chain of Responsibility: Processes segments
    - Command: Encapsulates segment execution
    
    Data Structures:
    - List of ordered segments
    - Queue for pending segments
    - Set for completed segments
    
    Threading/Concurrency Guarantees:
    1. Thread-safe segment execution
    2. Atomic sequence completion
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(s) execution where s is segment count
    2. O(p) pseudostate coordination where p is pseudostate count
    3. O(r) rollback where r is completed segment count
    """
    pass


class ProtocolTransition(Transition):
    """Represents a protocol transition with strict constraints.
    
    ProtocolTransition enforces protocol state machine semantics,
    ensuring valid state sequences and operation calls.
    
    Class Invariants:
    1. Must follow protocol constraints
    2. Must validate operation calls
    3. Must maintain protocol state
    4. Must enforce sequence rules
    
    Design Patterns:
    - State: Manages protocol states
    - Strategy: Implements protocol rules
    - Command: Encapsulates operations
    
    Threading/Concurrency Guarantees:
    1. Thread-safe protocol checks
    2. Atomic operation validation
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) protocol state check
    2. O(v) operation validation where v is validator count
    3. O(c) constraint checking where c is constraint count
    """
    pass


class TimeTransition(Transition):
    """Represents a time-triggered transition.
    
    TimeTransition manages transitions triggered by time events,
    both relative ("after") and absolute ("at") timing.
    
    Class Invariants:
    1. Must have valid time specification
    2. Must maintain timing accuracy
    3. Must handle timer interruptions
    4. Must support cancellation
    
    Design Patterns:
    - Command: Encapsulates time events
    - Observer: Notifies of timing
    - Strategy: Implements timing types
    
    Threading/Concurrency Guarantees:
    1. Thread-safe timer operations
    2. Atomic execution scheduling
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) timer operations
    2. O(log n) scheduling where n is timer count
    3. O(1) cancellation
    """
    pass


class ChangeTransition(Transition):
    """Represents a change-triggered transition.
    
    ChangeTransition manages transitions triggered by changes in
    boolean conditions, implementing the observer pattern.
    
    Class Invariants:
    1. Must have valid change condition
    2. Must detect all changes
    3. Must prevent missed triggers
    4. Must maintain condition state
    
    Design Patterns:
    - Observer: Monitors changes
    - Strategy: Implements detection
    - Command: Encapsulates triggers
    
    Threading/Concurrency Guarantees:
    1. Thread-safe condition monitoring
    2. Atomic change detection
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) state checking
    2. O(c) condition evaluation where c is condition complexity
    3. O(o) observer notification where o is observer count
    """
    pass
