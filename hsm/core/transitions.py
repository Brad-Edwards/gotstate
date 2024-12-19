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
        source: StateID,
        target: StateID,
        guard: Optional[AbstractGuard] = None,
        actions: Optional[List[AbstractAction]] = None,
        priority: int = 0,
    ) -> None:
        """
        Initialize a new transition.

        Args:
            source: The source state ID.
            target: The target state ID.
            guard: Optional condition for the transition.
            actions: Optional list of actions to execute.
            priority: Priority for conflict resolution (default: 0)

        Raises:
            TypeError: If source or target is None, or if parameters are of wrong type
            ValueError: If source or target is empty or whitespace
        """
        # Validate state IDs
        if source is None or target is None:
            raise TypeError("Source and target state IDs cannot be None")
        if not isinstance(source, str) or not isinstance(target, str):
            raise TypeError("Source and target must be strings")

        # Validate priority type before any string operations
        if not isinstance(priority, int) or isinstance(priority, bool):  # explicitly check for bool
            raise TypeError("Priority must be an integer")

        # Check for whitespace in state IDs
        if any(c.isspace() for c in source) or any(c.isspace() for c in target):
            raise ValueError("State IDs cannot contain whitespace")

        # Strip and check if empty
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError("Source and target state IDs cannot be empty or whitespace")

        # Validate guard type
        if guard is not None and not isinstance(guard, AbstractGuard):
            raise TypeError("Guard must be an instance of AbstractGuard")

        # Validate actions type and contents
        if actions is not None:
            if not isinstance(actions, list):
                raise TypeError("Actions must be a list")
            if not all(isinstance(action, AbstractAction) for action in actions):
                raise TypeError("All actions must be instances of AbstractAction")

        self._source = source
        self._target = target
        self._guard = guard
        # Create a new list to ensure immutability
        self._actions = list(actions) if actions is not None else []
        self._priority = priority

    def get_source_state_id(self) -> StateID:
        """Get the source state ID."""
        return self._source

    def get_target_state_id(self) -> StateID:
        """Get the target state ID."""
        return self._target

    def get_guard(self) -> Optional[AbstractGuard]:
        """Get the guard condition, if any."""
        return self._guard

    def get_actions(self) -> List[AbstractAction]:
        """Get the list of actions to execute."""
        # Return a copy of the actions list to maintain immutability
        return list(self._actions)

    def get_priority(self) -> int:
        """Get the transition priority."""
        return self._priority

    def __repr__(self) -> str:
        """Return string representation of the transition."""
        return (
            f"Transition(source='{self._source}', target='{self._target}', "
            f"priority={self._priority}, guard={'present' if self._guard else 'none'}, "
            f"actions={len(self._actions)})"
        )

    def __str__(self) -> str:
        """Return string representation of the transition."""
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        """Compare two transitions for equality."""
        if not isinstance(other, Transition):
            return False
        return (
            self._source == other._source
            and self._target == other._target
            and self._priority == other._priority
            and self._guard == other._guard
            and len(self._actions) == len(other._actions)
            and all(a1 == a2 for a1, a2 in zip(self._actions, other._actions))
        )

    def __hash__(self) -> int:
        """Generate hash for the transition."""
        return hash(
            (
                self._source,
                self._target,
                self._priority,
                self._guard,
                tuple(self._actions),  # Convert list to tuple for hashing
            )
        )
