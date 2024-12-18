# hsm/interfaces/abc.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

from hsm.interfaces.types import EventID, StateID, ValidationResult

# Type variables for generic protocols
T = TypeVar("T")
S = TypeVar("S", bound="AbstractState")


@runtime_checkable
class AbstractEvent(Protocol):
    """
    Protocol for events.

    Runtime Invariants:
    - Events are immutable after creation
    - Event IDs are unique within a session
    """

    def get_id(self) -> EventID: ...

    def get_payload(self) -> Any: ...

    def get_priority(self) -> int: ...


# Runtime checkable protocols
@runtime_checkable
class AbstractState(Protocol):
    """
    Protocol defining state behavior.

    Runtime Invariants:
    - State data is isolated
    - Entry/exit actions are atomic
    """

    @abstractmethod
    def on_entry(self, event: AbstractEvent, data: Any) -> None: ...

    @abstractmethod
    def on_exit(self, event: AbstractEvent, data: Any) -> None: ...

    def get_id(self) -> StateID: ...


@runtime_checkable
class AbstractStateMachine(Protocol):
    """
    Protocol defining the core state machine interface.

    Runtime Invariants:
    - Only one state is active at a time
    - State transitions are atomic
    - Event processing is sequential
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def process_event(self, event: "AbstractEvent") -> None: ...
    def get_current_state_id(self) -> StateID: ...
    def get_state(self) -> Optional[AbstractState]: ...


@runtime_checkable
class AbstractCompositeState(AbstractState, Protocol):
    """
    Protocol for hierarchical states.

    Runtime Invariants:
    - Substates form a valid tree structure
    - Initial state is always valid
    """

    @property
    def parent_state(self) -> Optional[AbstractState]: ...

    @parent_state.setter
    def parent_state(self, value: AbstractState) -> None: ...

    def get_substates(self) -> List[AbstractState]: ...

    def get_initial_state(self) -> AbstractState: ...

    def has_history(self) -> bool: ...

    def set_history_state(self, state: AbstractState) -> None: ...


@runtime_checkable
class AbstractGuard(Protocol):
    """
    Protocol for transition guards.

    Runtime Invariants:
    - Guards are stateless
    - Guard evaluation is deterministic
    """

    def check(self, event: AbstractEvent, state_data: Any) -> bool: ...


@runtime_checkable
class AbstractAction(Protocol):
    """
    Protocol for transition actions.

    Runtime Invariants:
    - Actions are atomic
    - Action execution order is preserved
    """

    def execute(self, event: AbstractEvent, state_data: Any) -> None: ...


@runtime_checkable
class AbstractHook(Protocol):
    """
    Protocol for state machine hooks.

    Runtime Invariants:
    - Hooks don't modify state machine behavior
    - Hook failures don't affect state machine operation
    """

    def on_enter(self, state_id: StateID) -> None: ...

    def on_exit(self, state_id: StateID) -> None: ...

    def pre_transition(self, transition: "AbstractTransition") -> None: ...

    def post_transition(self, transition: "AbstractTransition") -> None: ...


@runtime_checkable
class AbstractTransition(Protocol):
    """
    Protocol for state transitions.

    Runtime Invariants:
    - Source and target states exist
    - Guards and actions are valid
    """

    def get_source_state(self) -> AbstractState: ...

    def get_target_state(self) -> AbstractState: ...

    def get_guard(self) -> Optional[AbstractGuard]: ...

    def get_actions(self) -> List[AbstractAction]: ...

    def get_priority(self) -> int: ...


@runtime_checkable
class AbstractValidator(Protocol):
    """
    Protocol for validation operations.

    Runtime Invariants:
    - Validation rules are consistent
    - Results are deterministic
    """

    def validate_structure(self) -> List[ValidationResult]: ...

    def validate_behavior(self) -> List[ValidationResult]: ...

    def validate_data(self) -> List[ValidationResult]: ...


@runtime_checkable
class AbstractEventQueue(Protocol):
    """
    Protocol for event queues.

    Runtime Invariants:
    - FIFO order within same priority
    - Thread-safe operations
    """

    def enqueue(self, event: AbstractEvent) -> None: ...

    def dequeue(self) -> AbstractEvent: ...

    def is_full(self) -> bool: ...

    def is_empty(self) -> bool: ...


@runtime_checkable
class AbstractTimer(Protocol):
    """
    Protocol for timer operations.

    Runtime Invariants:
    - Timers are cancelable
    - Timer events maintain order
    """

    def schedule_timeout(self, duration: float, event: AbstractEvent) -> None: ...

    def cancel_timeout(self, event_id: EventID) -> None: ...


# Helper types for type checking
StateType = TypeVar("StateType", bound=AbstractState)
EventType = TypeVar("EventType", bound=AbstractEvent)
