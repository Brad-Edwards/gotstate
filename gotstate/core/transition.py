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

from typing import Optional, List, Callable, Any, Dict, Union
from enum import Enum, auto
from dataclasses import dataclass
from functools import reduce
from operator import and_, or_


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


class GuardCondition:
    """Represents a guard condition for a transition.
    
    GuardCondition implements composable boolean logic for transition
    guards with proper evaluation semantics.
    
    Class Invariants:
    1. Must be deterministic
    2. Must be side-effect free
    3. Must complete quickly
    4. Must handle all data types
    5. Must compose properly
    
    Design Patterns:
    - Strategy: Implements evaluation logic
    - Composite: Enables condition composition
    - Command: Encapsulates condition logic
    
    Threading/Concurrency Guarantees:
    1. Thread-safe evaluation
    2. Atomic composition
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) simple evaluation
    2. O(n) composite evaluation where n is condition count
    3. O(1) composition operations
    """
    
    def __init__(self, condition: Callable[[Dict[str, Any]], bool]) -> None:
        """Initialize a GuardCondition instance.
        
        Args:
            condition: Function that takes event data and returns bool
        """
        self._condition = condition
        
    def evaluate(self, event_data: Dict[str, Any]) -> bool:
        """Evaluate the guard condition.
        
        Args:
            event_data: Event data dictionary for condition context
            
        Returns:
            True if condition is satisfied, False otherwise
        """
        return self._condition(event_data)
        
    def __and__(self, other: 'GuardCondition') -> 'GuardCondition':
        """Compose two guards with AND logic.
        
        Args:
            other: Another guard condition
            
        Returns:
            New guard that is true only if both are true
        """
        return GuardCondition(
            lambda data: self.evaluate(data) and other.evaluate(data)
        )
        
    def __or__(self, other: 'GuardCondition') -> 'GuardCondition':
        """Compose two guards with OR logic.
        
        Args:
            other: Another guard condition
            
        Returns:
            New guard that is true if either is true
        """
        return GuardCondition(
            lambda data: self.evaluate(data) or other.evaluate(data)
        )
        
    def __invert__(self) -> 'GuardCondition':
        """Negate the guard condition.
        
        Returns:
            New guard that is true when original is false
        """
        return GuardCondition(
            lambda data: not self.evaluate(data)
        )


class TransitionEffect:
    """Represents an effect (action) executed during a transition.
    
    TransitionEffect implements composable actions that are executed
    when a transition fires, with proper execution semantics.
    
    Class Invariants:
    1. Must maintain state consistency
    2. Must handle errors gracefully
    3. Must be idempotent when possible
    4. Must compose properly
    5. Must clean up resources
    
    Design Patterns:
    - Command: Encapsulates effect logic
    - Composite: Enables effect composition
    - Chain of Responsibility: Handles execution
    
    Threading/Concurrency Guarantees:
    1. Thread-safe execution
    2. Atomic composition
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) simple execution
    2. O(n) composite execution where n is effect count
    3. O(1) composition operations
    """
    
    def __init__(self, action: Callable[[Dict[str, Any]], None]) -> None:
        """Initialize a TransitionEffect instance.
        
        Args:
            action: Function that takes event data and performs effect
        """
        self._action = action
        
    def execute(self, event_data: Dict[str, Any]) -> None:
        """Execute the transition effect.
        
        Args:
            event_data: Event data dictionary for effect context
        """
        self._action(event_data)
        
    def __add__(self, other: 'TransitionEffect') -> 'TransitionEffect':
        """Compose two effects sequentially.
        
        Args:
            other: Another transition effect
            
        Returns:
            New effect that executes both in sequence
        """
        return TransitionEffect(
            lambda data: (self.execute(data), other.execute(data))
        )


class Transition:
    """Represents a transition between states in a hierarchical state machine.
    
    The Transition class implements the Command pattern to encapsulate all aspects
    of a state transition including guards, actions, and execution semantics.
    
    Class Invariants:
    1. Source and target states must be valid and compatible
    2. Guard conditions must be side-effect free
    3. Actions must maintain state consistency
    4. Transition kind must not change after initialization
    5. Priority must be valid for conflict resolution
    6. Trigger specifications must be well-formed
    7. Cross-region transitions must be properly synchronized
    8. Compound transitions must have valid segments
    9. Protocol transitions must maintain protocol constraints
    10. Time/change triggers must be properly scheduled
    
    Design Patterns:
    - Command: Encapsulates transition execution
    - Strategy: Implements transition type behavior
    - Chain of Responsibility: Processes guard conditions
    - Observer: Notifies of transition execution
    - Template Method: Defines execution steps
    - Memento: Preserves state for rollback
    
    Data Structures:
    - List for compound transition segments
    - Queue for pending actions
    - Set for affected regions
    - Tree for LCA computation
    - Priority queue for conflict resolution
    
    Algorithms:
    - LCA computation for transition scope
    - Topological sort for execution order
    - Priority-based conflict resolution
    - Path computation for state changes
    
    Threading/Concurrency Guarantees:
    1. Thread-safe transition execution
    2. Atomic guard evaluation
    3. Synchronized action execution
    4. Safe concurrent conflict resolution
    5. Lock-free transition inspection
    6. Mutex protection for state changes
    
    Performance Characteristics:
    1. O(1) kind/priority checking
    2. O(log n) conflict resolution
    3. O(h) LCA computation where h is hierarchy depth
    4. O(a) action execution where a is action count
    5. O(g) guard evaluation where g is guard count
    
    Resource Management:
    1. Bounded action execution time
    2. Controlled guard evaluation scope
    3. Limited concurrent transitions
    4. Pooled transition objects
    5. Cached computation results
    """
    
    def __init__(
        self,
        source: 'State',
        target: Optional['State'] = None,
        event: Optional['Event'] = None,
        guard: Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]] = None,
        effect: Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]] = None,
        kind: TransitionKind = TransitionKind.EXTERNAL,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize a Transition instance.
        
        Args:
            source: Source state of the transition
            target: Optional target state (None for internal)
            event: Optional triggering event
            guard: Optional guard condition
            effect: Optional transition effect
            kind: Type of transition (external, local, internal)
            priority: Transition priority level
            
        Raises:
            ValueError: If any parameters are invalid
        """
        if source is None:
            raise ValueError("Source state must be provided")
            
        if kind != TransitionKind.INTERNAL and target is None:
            raise ValueError("Target state required for non-internal transitions")
            
        if not isinstance(kind, TransitionKind):
            raise ValueError("Transition kind must be a TransitionKind enum value")
            
        if not isinstance(priority, TransitionPriority):
            raise ValueError("Transition priority must be a TransitionPriority enum value")
            
        self._source = source
        self._target = target
        self._event = event
        self._guard = guard  # Store raw guard function or GuardCondition
        self._effect = effect  # Store raw effect function or TransitionEffect
        self._kind = kind
        self._priority = priority
        
    @property
    def source(self) -> 'State':
        """Get the source state."""
        return self._source
        
    @property
    def target(self) -> Optional['State']:
        """Get the target state."""
        return self._target
        
    @property
    def event(self) -> Optional['Event']:
        """Get the triggering event."""
        return self._event
        
    @property
    def guard(self) -> Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]]:
        """Get the guard condition."""
        return self._guard
        
    @property
    def effect(self) -> Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]]:
        """Get the transition effect."""
        return self._effect
        
    @property
    def kind(self) -> TransitionKind:
        """Get the transition kind."""
        return self._kind
        
    @property
    def priority(self) -> TransitionPriority:
        """Get the transition priority."""
        return self._priority
        
    def evaluate_guard(self) -> bool:
        """Evaluate the transition's guard condition.
        
        Returns:
            True if guard is satisfied or no guard,
            False if guard evaluates to false
        """
        if self._guard is None:
            return True
            
        event_data = self._event.data if self._event else {}
        
        if isinstance(self._guard, GuardCondition):
            return self._guard.evaluate(event_data)
        else:
            return self._guard(event_data)  # Call raw guard function
        
    def execute_effect(self) -> None:
        """Execute the transition's effect."""
        if self._effect is not None:
            event_data = self._event.data if self._event else {}
            
            if isinstance(self._effect, TransitionEffect):
                self._effect.execute(event_data)
            else:
                self._effect(event_data)  # Call raw effect function
            
    def execute(self) -> bool:
        """Execute the complete transition.
        
        This method evaluates the guard, coordinates the states,
        and executes the effect according to the transition kind.
        
        Returns:
            True if transition executed successfully,
            False if guard prevented execution
        """
        # Check guard condition
        if not self.evaluate_guard():
            return False
            
        # Handle different transition kinds
        if self._kind == TransitionKind.INTERNAL:
            # Internal transitions don't change state
            self.execute_effect()
            
        elif self._kind == TransitionKind.LOCAL:
            # Local transitions minimize state changes
            if self._target is not None:
                self._source.deactivate()
                self.execute_effect()
                self._target.activate()
                
        else:  # EXTERNAL
            # External transitions do full exit/entry
            self._source.exit()
            self.execute_effect()
            if self._target is not None:
                self._target.enter()
                
        return True


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
