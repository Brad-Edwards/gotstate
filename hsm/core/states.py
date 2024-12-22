# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from hsm.core.base import StateBase
from hsm.core.errors import TransitionError, ValidationError
from hsm.core.events import Event
from hsm.core.transitions import Transition

if TYPE_CHECKING:
    from hsm.core.hooks import Hook
    from hsm.core.validations import Validator


class State(StateBase):
    """
    Represents a state in the state machine. Manages state-specific behavior
    including entry/exit actions and local data.

    This version does NOT store parent references or child references directly,
    assuming that the 'graph.py' infrastructure handles hierarchy.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize a state with its name and optional actions.

        :param name: Name identifying this state within its parent scope.
        :param entry_actions: Actions executed upon entering this state.
        :param exit_actions: Actions executed upon exiting this state.
        """
        super().__init__(name=name, entry_actions=entry_actions or [], exit_actions=exit_actions or [])
        self.data: Dict[str, Any] = {}


class CompositeState(StateBase):
    """
    A composite state that can contain other states. This version does NOT
    store children or track them directly, leaving that to the StateGraph.
    If you wish to store them here, see the commented approach below.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize a composite state.

        :param name: Name identifying this state.
        :param entry_actions: Actions executed upon entering this state.
        :param exit_actions: Actions executed upon exiting this state.
        """
        super().__init__(name=name, entry_actions=entry_actions or [], exit_actions=exit_actions or [])
        # This state can contain children, but we rely on `graph.py` for storing them.
        # We'll keep an _initial_state reference if desired, but typically that's also in the graph.
        self._initial_state: Optional[State] = None

    @property
    def initial_state(self) -> Optional[State]:
        """Optional: get the initial state. If you're using StateGraph, use set_initial_state there."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Optional[State]) -> None:
        """
        Optional: set the initial state. In a graph-based design, you'd call
        `StateGraph.set_initial_state(self, state)` instead.
        """
        self._initial_state = state
