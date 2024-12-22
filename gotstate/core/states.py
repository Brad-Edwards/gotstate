# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from gotstate.core.base import StateBase


class State(StateBase):
    """
    Represents a state in the state machine. Manages state-specific behavior
    including entry/exit actions. All relationships and data are managed through StateGraph.
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def data(self) -> Dict[str, Any]:
        """
        Access state data through StateGraph. This property exists for compatibility
        but will raise an error if accessed before the state is added to a graph.
        """
        raise AttributeError(
            "State data cannot be accessed directly. Use StateGraph's get_state_data/set_state_data methods"
        )


class CompositeState(State):
    """
    A state that can contain other states. All hierarchy management is handled
    through StateGraph.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        super().__init__(name, entry_actions, exit_actions)

    @property
    def initial_state(self) -> Optional[State]:
        """Get the initial state through the graph."""
        if hasattr(self, "_graph"):
            return self._graph.get_initial_state(self)
        return None
