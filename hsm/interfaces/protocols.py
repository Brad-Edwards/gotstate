# hsm/interfaces/protocols.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import Any, List, Optional, Protocol, runtime_checkable

from hsm.interfaces.abc import AbstractAction, AbstractGuard
from hsm.interfaces.types import EventID, StateID


@runtime_checkable
class Event(Protocol):
    """
    Event protocol for type checking.

    Methods:
        get_id(): Returns a unique event ID (EventID).
        get_payload(): Returns the event payload, may be None.
        get_priority(): Returns the priority (int), with 0 as default.

    Runtime Invariants:
    - Events are immutable after creation.
    - Event IDs are unique within a session.

    Error Handling:
    - Implementations must ensure no exceptions occur during getter calls.
      If exceptions occur, they are considered implementation errors.
    """

    def get_id(self) -> EventID:
        """Get the unique event identifier."""
        ...

    def get_payload(self) -> Any:
        """Get the event payload, which can be any object or None."""
        ...

    def get_priority(self) -> int:
        """Get the event priority. Lower numbers may mean higher priority."""
        ...


@runtime_checkable
class Transition(Protocol):
    """
    Transition protocol for type checking.

    Methods:
        get_source_state_id(): Returns the source state's ID.
        get_target_state_id(): Returns the target state's ID.
        get_guard(): Returns an optional guard implementing AbstractGuard.
        get_actions(): Returns a list of actions implementing AbstractAction.
        get_priority(): Returns an integer priority for conflict resolution.

    Runtime Invariants:
    - Source and target states exist.
    - Guards and actions are valid and follow their respective protocols.
    - Priority determines which transition is chosen if multiple are enabled.

    Error Handling:
    - Guard or Action failures are raised as HSMErrors at runtime by the caller.
      The Transition itself should not raise errors on property access.
    """

    def get_source_state_id(self) -> StateID:
        """Return the ID of the source state for this transition."""
        ...

    def get_target_state_id(self) -> StateID:
        """Return the ID of the target state for this transition."""
        ...

    def get_guard(self) -> Optional[AbstractGuard]:
        """
        Return the guard for this transition, or None if no guard is specified.

        If a guard is present, its check() method is called before the transition
        is taken. If it returns False, the transition is not taken.
        """
        ...

    def get_actions(self) -> List[AbstractAction]:
        """
        Return the list of actions for this transition.

        Actions are executed atomically and in order if the guard passes.
        """
        ...

    def get_priority(self) -> int:
        """
        Return the priority of this transition.

        Higher priority transitions are chosen first if multiple transitions
        are available from the same state in response to the same event.
        """
        ...
