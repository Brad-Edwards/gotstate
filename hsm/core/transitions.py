# hsm/core/transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import List, Optional

from hsm.interfaces.abc import AbstractAction, AbstractGuard, AbstractState, AbstractTransition
from hsm.interfaces.types import StateID


class Transition(AbstractTransition):
    """
    Concrete implementation of a state machine transition.

    A transition represents a possible change from one state to another,
    optionally guarded by a condition and executing actions when taken.

    Runtime Invariants:
    - Source and target states are immutable after creation
    - Guards and actions are immutable after creation
    - Priority is immutable after creation

    Attributes:
        _source: The source state.
        _target: The target state.
        _guard: Optional condition that must be true for transition to be taken.
        _actions: List of actions to execute when transition is taken.
        _priority: Integer priority for conflict resolution.
    """

    def __init__(
        self,
        source: AbstractState,
        target: AbstractState,
        guard: Optional[AbstractGuard] = None,
        actions: Optional[List[AbstractAction]] = None,
        priority: int = 0,
    ) -> None:
        """
        Initialize a new transition.

        Args:
            source: The source state.
            target: The target state.
            guard: Optional condition for the transition.
            actions: Optional list of actions to execute.
            priority: Priority for conflict resolution (default: 0)

        Raises:
            TypeError: If source or target is None
        """
        if source is None or target is None:
            raise TypeError("Source and target states cannot be None")
        self._source = source
        self._target = target
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state(self) -> AbstractState:
        """Get the source state."""
        return self._source

    def get_target_state(self) -> AbstractState:
        """Get the target state."""
        return self._target

    def get_guard(self) -> Optional[AbstractGuard]:
        """Get the guard condition, if any."""
        return self._guard

    def get_actions(self) -> List[AbstractAction]:
        """Get the list of actions to execute."""
        return self._actions

    def get_priority(self) -> int:
        """Get the transition priority."""
        return self._priority

    def __repr__(self) -> str:
        """Return string representation of the transition."""
        return (
            f"Transition(source='{self._source.get_id()}', target='{self._target.get_id()}', "
            f"priority={self._priority}, guard={'present' if self._guard else 'none'}, "
            f"actions={len(self._actions)})"
        )
