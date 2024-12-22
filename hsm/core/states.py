# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set

from hsm.core.base import StateBase


class State(StateBase):
    """
    Represents a state in the state machine. Manages state-specific behavior
    including entry/exit actions.

    While state relationships and data are managed through StateGraph,
    some attributes are maintained for backward compatibility.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        """Initialize a state with name and optional actions."""
        self.name = name
        self.entry_actions = entry_actions or []
        self.exit_actions = exit_actions or []
        # Maintained for backward compatibility
        self.data: Dict[str, Any] = {}
        self.parent: Optional[State] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class CompositeState(State):
    """
    A state that can contain other states. While hierarchy is managed through
    StateGraph, some attributes are maintained for backward compatibility.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        super().__init__(name, entry_actions, exit_actions)
        # Maintained for backward compatibility
        self._children: Set[State] = set()
        self._initial_state: Optional[State] = None

    @property
    def initial_state(self) -> Optional[State]:
        """Get the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Optional[State]) -> None:
        """Set the initial state."""
        self._initial_state = state
