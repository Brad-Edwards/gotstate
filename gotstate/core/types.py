"""
Type definitions and enums for the state machine.

This module contains shared type definitions and enums used across
the state machine implementation. It helps break circular dependencies
between modules and provides a central location for type information.

Design:
- No runtime dependencies on other modules
- Only contains type definitions and enums
- Used by both state.py and transition.py
- Provides type hints for static analysis
"""

from enum import Enum, auto
from typing import TypeVar, Dict, Any, Optional, Union, Callable


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


# Type variables for generic type hints
State = TypeVar('State', bound='BaseState')
Transition = TypeVar('Transition', bound='BaseTransition')
Region = TypeVar('Region', bound='BaseRegion')

# Type aliases for common types
StateData = Dict[str, Any]
GuardFunction = Callable[[Dict[str, Any]], bool]
EffectFunction = Callable[[Dict[str, Any]], None] 