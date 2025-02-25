"""Common type definitions for gotstate."""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import TypeAlias, Protocol

# Basic ID types
StateId = NewType("StateId", str)
EventId = NewType("EventId", str)
TransitionId = NewType("TransitionId", str)
RegionId = NewType("RegionId", str)
GuardId = NewType("GuardId", str)
ActionId = NewType("ActionId", str)

# Type for event data
EventData: TypeAlias = Any

# Protocol for callable objects
class Callable_Protocol(Protocol):
    """Protocol for callable objects."""
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

# Action function signatures
EventHandler: TypeAlias = Callable[[EventId, EventData], None]
StateAction: TypeAlias = Callable[[EventId, EventData], None]
TransitionAction: TypeAlias = Callable[[EventId, EventData], None]
GuardFunction: TypeAlias = Callable[[EventId, EventData], bool]

# Action functions for async operations
AsyncEventHandler: TypeAlias = Callable[[EventId, EventData], Any]
AsyncStateAction: TypeAlias = Callable[[EventId, EventData], Any]
AsyncTransitionAction: TypeAlias = Callable[[EventId, EventData], Any]
AsyncGuardFunction: TypeAlias = Callable[[EventId, EventData], Any]

# Generic state types for internal type hinting
S = TypeVar('S')
T = TypeVar('T')
E = TypeVar('E')

# Event queue types
EventQueueEntry: TypeAlias = Tuple[EventId, EventData]
EventQueue: TypeAlias = List[EventQueueEntry] 