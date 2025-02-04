"""
Core package providing the fundamental HFSM functionality.

Architecture:
- Implements hierarchical state machine core components
- Manages state hierarchy, transitions, events, and regions
- Coordinates between components through machine orchestration

Design Patterns:
- Composite Pattern for state hierarchy
- Observer Pattern for state changes
- Command Pattern for transitions
- Strategy Pattern for event processing
- Mediator Pattern for coordination

Security:
- Input validation at module boundaries
- State data isolation
- Resource usage monitoring
- Type system safety checks

Cross-cutting:
- Error handling with consistent propagation
- Performance optimization for state operations
- Monitoring hooks for metrics
- Testing boundaries for validation
"""

# Import order matters to avoid circular dependencies
from .state import State, StateType, CompositeState, PseudoState, HistoryState
from .event import (
    Event,
    EventKind,
    EventPriority,
    SignalEvent,
    CallEvent,
    TimeEvent,
    ChangeEvent,
    CompletionEvent,
    EventQueue,
)
from .transition import Transition
from .region import Region
from .machine.state_machine import StateMachine

__all__ = [
    # State classes
    "State",
    "StateType",
    "CompositeState",
    "PseudoState",
    "HistoryState",
    # Event classes
    "Event",
    "EventKind",
    "EventPriority",
    "SignalEvent",
    "CallEvent",
    "TimeEvent",
    "ChangeEvent",
    "CompletionEvent",
    "EventQueue",
    # Other core classes
    "Transition",
    "Region",
    "StateMachine",
]
