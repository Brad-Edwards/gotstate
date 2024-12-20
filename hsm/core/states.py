# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from hsm.core.base import StateBase
from hsm.core.events import Event
from hsm.core.transitions import Transition

if TYPE_CHECKING:
    from hsm.core.hooks import Hook
    from hsm.core.validations import Validator


class State(StateBase):
    """
    Represents a state in the state machine. Manages state-specific behavior
    including entry/exit actions and local data.
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


class CompositeState(State):
    """
    A state that can contain other states, forming a hierarchy.
    """

    def __init__(
        self,
        name: str,
        initial_state: Optional[State] = None,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize a composite state.

        :param name: Name identifying this state.
        :param initial_state: The default state to enter when this composite state is entered.
        :param entry_actions: Actions executed upon entering this state.
        :param exit_actions: Actions executed upon exiting this state.
        """
        super().__init__(name, entry_actions, exit_actions)
        self._children = set()
        self._initial_state = None
        # Set initial state through property to ensure proper parent-child relationship
        if initial_state:
            self.initial_state = initial_state

    @property
    def initial_state(self) -> Optional[State]:
        """Get the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Optional[State]) -> None:
        """Set the initial state and establish parent-child relationship."""
        if state:
            state.parent = self
            self._children.add(state)
        self._initial_state = state

    def add_child_state(self, state: State) -> None:
        """Add a child state to this composite state."""
        state.parent = self
        self._children.add(state)

    def get_child_state(self, name: str) -> Optional[State]:
        """Get a child state by name."""
        for child in self._children:
            if child.name == name:
                return child
        return None

    def get_children(self) -> List[State]:
        """Get all child states."""
        return list(self._children)


class StateMachine:
    def __init__(self, initial_state: State, validator: Optional[Validator] = None, hooks: Optional[List[Hook]] = None):
        self._initial_state = initial_state
        self._current_state = initial_state  # Set initial state immediately
        self._validator = validator or Validator()
        self._hooks = hooks or []
        self._started = False

    def get_current_state(self) -> Optional[State]:
        return self._current_state

    def process_event(self, event: Event) -> None:
        try:
            transition = self._get_transition(event)
            if transition:
                self._execute_transition(transition, event)
        except TransitionError as e:
            if self._error_recovery:
                self._error_recovery.recover(e, self)
            else:
                raise

    def _execute_transition(self, transition: Transition, event: Event) -> None:
        try:
            transition.execute_actions(event)
            self._current_state = transition.target
        except Exception as e:
            # Wrap any action execution errors
            raise TransitionError(f"Action execution failed: {str(e)}") from e
