# hsm/core/transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import List, Optional

from hsm.interfaces.abc import AbstractAction, AbstractGuard, AbstractTransition
from hsm.interfaces.types import StateID


class Transition(AbstractTransition):
    """
    Concrete implementation of a state machine transition.

    A transition represents a possible change from one state to another,
    optionally guarded by a condition and executing actions when taken.

    Runtime Invariants:
    - Source and target state IDs are immutable after creation
    - Guards and actions are immutable after creation
    - Priority is immutable after creation
    - Empty or whitespace state IDs are not allowed

    Attributes:
        _source_id: ID of the source state
        _target_id: ID of the target state
        _guard: Optional condition that must be true for transition to be taken
        _actions: List of actions to execute when transition is taken
        _priority: Integer priority for conflict resolution
    """

    def __init__(
        self,
        source_id: StateID,
        target_id: StateID,
        guard: Optional[AbstractGuard] = None,
        actions: Optional[List[AbstractAction]] = None,
        priority: int = 0,
    ) -> None:
        """
        Initialize a new transition.

        Args:
            source_id: ID of the source state
            target_id: ID of the target state
            guard: Optional condition for the transition
            actions: Optional list of actions to execute
            priority: Priority for conflict resolution (default: 0)

        Raises:
            ValueError: If source_id or target_id is empty or whitespace
            TypeError: If source_id or target_id is None
        """
        if source_id is None or target_id is None:
            raise TypeError("State IDs cannot be None")
        if not source_id.strip() or not target_id.strip():
            raise ValueError("State IDs cannot be empty or whitespace")

        self._source_id = source_id
        self._target_id = target_id
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state_id(self) -> StateID:
        """Get the source state ID."""
        return self._source_id

    def get_target_state_id(self) -> StateID:
        """Get the target state ID."""
        return self._target_id

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
            f"Transition(source='{self._source_id}', target='{self._target_id}', "
            f"priority={self._priority}, guard={'present' if self._guard else 'none'}, "
            f"actions={len(self._actions)})"
        )
