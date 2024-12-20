# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

from hsm.core.base import StateBase

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
        super().__init__(name=name,
                         entry_actions=entry_actions or [],
                         exit_actions=exit_actions or [])


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
        self._children: Dict[str, State] = {}
        self.initial_state = initial_state

    def add_child_state(self, state: State) -> None:
        """Add a child state to this composite state."""
        if state.name in self._children:
            raise ValueError(f"State '{state.name}' already exists")
        self._children[state.name] = state
        state.parent = self

    def get_child_state(self, name: str) -> Optional[State]:
        """Get a child state by name"""
        return self._children.get(name)

    def get_children(self) -> List[State]:
        """Get all child states."""
        return list(self._children.values())

    @property
    def initial_state(self) -> Optional[State]:
        """Get the initial state."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: State) -> None:
        """Set the initial state."""
        if state not in self._children and state is not None:
            self.add_child_state(state)
        self._initial_state = state


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
