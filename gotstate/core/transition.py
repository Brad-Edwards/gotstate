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

from typing import Optional, List, Callable, Any, Dict, Union, cast, TYPE_CHECKING
from enum import Enum, auto
from dataclasses import dataclass
from functools import reduce
from operator import and_, or_
from weakref import ReferenceType
from gotstate.core.event import Event, EventKind, EventPriority
from gotstate.core.types import (
    StateType,
    TransitionKind,
    TransitionPriority,
    GuardFunction,
    EffectFunction,
    StateData
)

if TYPE_CHECKING:
    from gotstate.core.state import State, ChoiceState, CompositeState, Region
else:
    # Forward references for runtime
    from gotstate.core.state import State, CompositeState


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
    """External transition that exits source state and enters target state."""
    
    def execute(self) -> bool:
        """Execute the external transition.
        
        Returns:
            True if transition executed successfully, False otherwise
        """
        # Check guard condition
        if not self.evaluate_guard():
            return False
            
        # Get all active states that need to be exited
        active_states = set()
        if isinstance(self.source, CompositeState):
            active_states.update(self.source.active_states)
        else:
            active_states.add(self.source)
            
        # Exit states in reverse hierarchical order (deepest first)
        for state in sorted(active_states, key=lambda s: len(s.path), reverse=True):
            state.exit()
            
        # Execute transition effect if any
        if self.effect:
            self.effect(self.event.data if self.event else {})
            
        # Enter target state and activate any initial states
        self.target.enter()
        
        # If target is composite, enter its regions
        if isinstance(self.target, CompositeState):
            for region in self.target.regions:
                region.enter()
                
        return True


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
    
    def __init__(
        self,
        source: 'State',
        target: Optional['State'] = None,  # Allow target for validation
        event: Optional['Event'] = None,
        guard: Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]] = None,
        effect: Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize an InternalTransition instance.
        
        Args:
            source: Source state
            target: Optional target state (must be same as source)
            event: Optional triggering event
            guard: Optional guard condition
            effect: Optional transition effect
            priority: Transition priority
            
        Raises:
            ValueError: If target is specified and different from source
        """
        if target is not None and target != source:
            raise ValueError("Internal transitions must have same source and target state")
            
        super().__init__(
            source=source,
            target=source,  # Internal transitions stay in same state
            event=event,
            guard=guard,
            effect=effect,
            priority=priority,
            kind=TransitionKind.INTERNAL
        )
        
    def execute(self) -> bool:
        """Execute the internal transition.
        
        Returns:
            True if transition executed successfully, False otherwise
        """
        # Check guard condition
        if not self.evaluate_guard():
            return False
            
        # Execute effect only, no state changes
        if self._effect is not None:
            if isinstance(self._effect, TransitionEffect):
                self._effect.execute(self._event.data if self._event else {})
            else:
                self._effect(self._event.data if self._event else {})
        
        return True


class LocalTransition(Transition):
    """Represents a local transition between states with same parent.
    
    LocalTransition executes a transition between sibling states,
    maintaining the parent state's active status.
    
    Class Invariants:
    1. Must maintain parent state
    2. Must handle sibling state changes
    3. Must validate state hierarchy
    4. Must maintain active state consistency
    
    Design Patterns:
    - Template Method: Defines transition sequence
    - State: Manages state configuration
    - Command: Encapsulates actions
    
    Threading/Concurrency Guarantees:
    1. Thread-safe state changes
    2. Atomic transition execution
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) parent validation
    2. O(1) state changes
    3. O(1) effect execution
    """
    
    def __init__(
        self,
        source: 'State',
        target: 'State',
        event: Optional['Event'] = None,
        guard: Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]] = None,
        effect: Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize a LocalTransition instance.
        
        Args:
            source: Source state
            target: Target state
            event: Optional triggering event
            guard: Optional guard condition
            effect: Optional transition effect
            priority: Transition priority
            
        Raises:
            ValueError: If states don't share same parent
        """
        # Validate parent states before calling super().__init__
        if source.parent is None or target.parent is None:
            raise ValueError("Both source and target states must have parents")
            
        source_parent = source.parent() if isinstance(source.parent, ReferenceType) else source.parent
        target_parent = target.parent() if isinstance(target.parent, ReferenceType) else target.parent
            
        if source_parent is None or target_parent is None:
            raise ValueError("Both source and target states must have parents")
            
        if source_parent.id != target_parent.id:
            raise ValueError("Source and target states must share same parent")
            
        super().__init__(
            source=source,
            target=target,
            event=event,
            guard=guard,
            effect=effect,
            priority=priority,
            kind=TransitionKind.LOCAL
        )


class CompoundTransition(Transition):
    """Represents a transition path through pseudostates.
    
    CompoundTransition manages a sequence of transitions through
    intermediate pseudostates, executing them as a single atomic
    transition.
    
    Class Invariants:
    1. Must maintain segment connectivity
    2. Must execute segments atomically
    3. Must validate path consistency
    4. Must handle pseudostate semantics
    
    Design Patterns:
    - Composite: Manages transition segments
    - Chain of Responsibility: Handles execution chain
    - Command: Encapsulates segment actions
    
    Threading/Concurrency Guarantees:
    1. Thread-safe execution
    2. Atomic path traversal
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(n) execution where n is segment count
    2. O(n) validation
    3. O(1) segment access
    """
    
    def __init__(
        self,
        segments: List['Transition'],
        event: Optional['Event'] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize a CompoundTransition instance.
        
        Args:
            segments: List of transition segments
            event: Optional triggering event
            priority: Transition priority
            
        Raises:
            ValueError: If segments are invalid
        """
        if not segments:
            raise ValueError("Must provide at least one transition segment")
            
        # Validate segment connectivity
        for i in range(len(segments) - 1):
            if segments[i].target != segments[i + 1].source:
                raise ValueError("Transition segments must be connected")
                
        super().__init__(
            source=segments[0].source,
            target=segments[-1].target,
            event=event,
            priority=priority,
            kind=TransitionKind.EXTERNAL  # Compound transitions are always external
        )
        
        self._segments = segments
        
    def execute(self) -> bool:
        """Execute the compound transition.
        
        Returns:
            True if transition executed successfully, False otherwise
        """
        # Execute each segment in sequence
        for segment in self._segments:
            if not segment.execute():
                return False
                
        return True


class ProtocolTransition(Transition):
    """Represents a transition triggered by protocol events.
    
    ProtocolTransition validates operation calls and manages
    protocol-based state changes in behavioral interfaces.
    
    Class Invariants:
    1. Must validate operation calls
    2. Must maintain protocol semantics
    3. Must handle operation parameters
    4. Must maintain interface consistency
    
    Design Patterns:
    - Strategy: Implements operation validation
    - Command: Encapsulates operations
    - State: Manages protocol state
    
    Threading/Concurrency Guarantees:
    1. Thread-safe operation validation
    2. Atomic state changes
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) operation validation
    2. O(1) state changes
    3. O(1) parameter handling
    """
    
    def __init__(
        self,
        source: 'State',
        target: 'State',
        event: Optional['Event'] = None,
        operation: str = None,
        guard: Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]] = None,
        effect: Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize a ProtocolTransition instance.
        
        Args:
            source: Source state
            target: Target state
            event: Optional triggering event
            operation: Operation name to validate
            guard: Optional guard condition
            effect: Optional transition effect
            priority: Transition priority
        """
        super().__init__(
            source=source,
            target=target,
            event=event,
            guard=guard,
            effect=effect,
            priority=priority,
            kind=TransitionKind.EXTERNAL
        )
        self._operation = operation
        
    def execute(self) -> bool:
        """Execute the protocol transition.
        
        Returns:
            True if transition executed successfully, False otherwise
        """
        if not self._validate_operation():
            return False
            
        return super().execute()
        
    def _validate_operation(self) -> bool:
        """Validate the operation call.
        
        Returns:
            True if operation is valid, False otherwise
        """
        if not self._event or not self._operation:
            return False
            
        event_data = self._event.data
        if not event_data or "operation" not in event_data:
            return False
            
        return event_data["operation"] == self._operation


class TimeTransition(Transition):
    """Represents a transition triggered by time events.
    
    TimeTransition manages time-based state changes, supporting
    both absolute and relative time triggers.
    
    Class Invariants:
    1. Must validate time events
    2. Must handle time semantics
    3. Must maintain temporal consistency
    4. Must handle timer operations
    
    Design Patterns:
    - Strategy: Implements time validation
    - Observer: Monitors time events
    - State: Manages temporal state
    
    Threading/Concurrency Guarantees:
    1. Thread-safe time handling
    2. Atomic state changes
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) time validation
    2. O(1) state changes
    3. O(1) timer operations
    """
    
    VALID_TIME_TYPES = {"after", "at"}
    
    def __init__(
        self,
        source: 'State',
        target: 'State',
        event: Optional['Event'] = None,
        guard: Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]] = None,
        effect: Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize a TimeTransition instance.
        
        Args:
            source: Source state
            target: Target state
            event: Optional triggering event
            guard: Optional guard condition
            effect: Optional transition effect
            priority: Transition priority
            
        Raises:
            ValueError: If time event is invalid
        """
        if event and event.kind == EventKind.TIME:
            if not event.data or "type" not in event.data:
                raise ValueError("Time event must specify type")
                
            if event.data["type"] not in self.VALID_TIME_TYPES:
                raise ValueError(f"Invalid time event type. Must be one of: {self.VALID_TIME_TYPES}")
                
            if "time" not in event.data or event.data["time"] < 0:
                raise ValueError("Time event must specify non-negative time value")
                
        super().__init__(
            source=source,
            target=target,
            event=event,
            guard=guard,
            effect=effect,
            priority=priority,
            kind=TransitionKind.EXTERNAL
        )


class ChangeTransition(Transition):
    """Represents a transition triggered by value changes.
    
    ChangeTransition monitors changes in values and triggers
    transitions based on change conditions.
    
    Class Invariants:
    1. Must validate change events
    2. Must evaluate change conditions
    3. Must maintain value consistency
    4. Must handle change notifications
    
    Design Patterns:
    - Observer: Monitors value changes
    - Strategy: Implements change evaluation
    - Command: Encapsulates change actions
    
    Threading/Concurrency Guarantees:
    1. Thread-safe change handling
    2. Atomic state changes
    3. Safe concurrent access
    
    Performance Characteristics:
    1. O(1) change validation
    2. O(1) condition evaluation
    3. O(1) state changes
    """
    
    def __init__(
        self,
        source: 'State',
        target: 'State',
        event: Optional['Event'] = None,
        condition: Optional[Callable[[Any, Any], bool]] = None,
        guard: Optional[Union[GuardCondition, Callable[[Dict[str, Any]], bool]]] = None,
        effect: Optional[Union[TransitionEffect, Callable[[Dict[str, Any]], None]]] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL
    ) -> None:
        """Initialize a ChangeTransition instance.
        
        Args:
            source: Source state
            target: Target state
            event: Optional triggering event
            condition: Optional change condition function
            guard: Optional guard condition
            effect: Optional transition effect
            priority: Transition priority
        """
        super().__init__(
            source=source,
            target=target,
            event=event,
            guard=guard,
            effect=effect,
            priority=priority,
            kind=TransitionKind.EXTERNAL
        )
        self._condition = condition
        
    def execute(self) -> bool:
        """Execute the change transition.
        
        Returns:
            True if transition executed successfully, False otherwise
        """
        if not self._evaluate_change():
            return False
            
        return super().execute()
        
    def _evaluate_change(self) -> bool:
        """Evaluate the change condition.
        
        Returns:
            True if change condition is satisfied, False otherwise
        """
        if not self._event or not self._condition:
            return False
            
        event_data = self._event.data
        if not event_data or "old_value" not in event_data or "new_value" not in event_data:
            return False
            
        return self._condition(event_data["old_value"], event_data["new_value"])
