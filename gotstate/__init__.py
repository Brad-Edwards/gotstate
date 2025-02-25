"""
GotState: A Pythonic UML-compliant Hierarchical Finite State Machine (HFSM) library.

This package provides a robust implementation of hierarchical state machines following
UML state machine semantics.

Features:
    - Complete UML State Machine compliance
    - Hierarchical state organization
    - Parallel region support
    - All UML pseudostates (history, choice, junction, etc.)
    - Parent state re-entry
    - Run-to-completion semantics
    - Asynchronous support
    - Thread-safe operations

Responsibilities:
    - State machine definition and execution
    - Event processing and transition management
    - Hierarchical state composition 
    - Parallel region handling
    - History state tracking
    - Runtime validation

Interactions:
    - Client code through public API
    - Python type system for static/runtime type checking
    - Operating system for concurrency primitives
    - Storage systems for persistence
    - Logging system for diagnostics

Cross-cutting Concerns:
    Thread Safety:
        - All public APIs are thread-safe
        - Internal state protected by appropriate locks
        - Documented thread safety guarantees per component

    Error Handling:
        - Structured error hierarchy
        - Consistent error reporting
        - Clean error recovery paths

    Logging:
        - Structured logging format
        - Configurable verbosity levels
        - Performance impact minimized

    Performance:
        - O(1) state lookup where possible
        - Bounded memory usage
        - Predictable latency

    Security:
        - Input validation on all public APIs
        - Safe serialization/deserialization
        - Protected internal state
"""

__version__ = "2.0.0"

# Core exports
from gotstate.core.state import State
from gotstate.core.statemachine import StateMachine
from gotstate.core.transition import Transition
from gotstate.core.event import Event
from gotstate.core.pseudostate import (
    PseudoState,
    InitialState,
    HistoryState,
    DeepHistoryState,
    ShallowHistoryState,
    ChoiceState,
    JunctionState,
    ForkState,
    JoinState,
    EntryPointState,
    ExitPointState,
    TerminateState,
)
from gotstate.core.region import Region
from gotstate.core.guard import Guard
from gotstate.core.action import Action

# Type exports
from gotstate.types.common import (
    StateId,
    EventId,
    TransitionId,
    RegionId,
)

# Extension exports
from gotstate.extensions.async_sm import AsyncStateMachine

# Exceptions
from gotstate.core.exceptions import (
    StateError,
    TransitionError,
    EventError,
    StateMachineError,
)
