# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from copy import deepcopy
from typing import Any, Dict, List, Optional

from hsm.interfaces.abc import AbstractCompositeState, AbstractEvent, AbstractState
from hsm.interfaces.types import StateID


class State(AbstractState):
    """
    Base class for state implementations.

    This class provides a foundation for implementing states in a hierarchical state machine.
    It manages state data and lifecycle events (enter/exit).

    Attributes:
        _state_id: Unique identifier for the state

    Runtime Invariants:
        - State ID is immutable after initialization
        - State data is isolated between states
        - Enter/exit calls are paired
    """

    def __init__(self, state_id: StateID):
        """
        Initialize a new state.

        Args:
            state_id: Unique identifier for this state

        Raises:
            ValueError: If state_id is empty or only whitespace
            TypeError: If state_id is None or not a string
        """
        if state_id is None:
            raise TypeError("state_id cannot be None")
        if not isinstance(state_id, str):
            raise TypeError("state_id must be a string")
        if not state_id or state_id.strip() == "":
            raise ValueError("state_id cannot be empty")

        self._state_id = state_id

    def on_entry(self, event: AbstractEvent, data: Any) -> None:
        """
        Called when the state is entered.

        This method should be overridden by concrete implementations to define
        entry behavior.

        Raises:
            NotImplementedError: If not overridden by concrete class
        """
        raise NotImplementedError("on_entry must be implemented by concrete state classes")

    def on_exit(self, event: AbstractEvent, data: Any) -> None:
        """
        Called when the state is exited.

        This method should be overridden by concrete implementations to define
        exit behavior.

        Raises:
            NotImplementedError: If not overridden by concrete class
        """
        raise NotImplementedError("on_exit must be implemented by concrete state classes")

    def get_id(self) -> StateID:
        """
        Get the state's unique identifier.

        Returns:
            The state's ID as a string
        """
        return self._state_id


class CompositeState(State, AbstractCompositeState):
    """
    A state that can contain other states, enabling hierarchical state machines.

    This class extends the base State class to support hierarchical state structures
    with substates and history tracking.

    Attributes:
        _substates: List of contained states
        _initial_state: Default state to enter when this composite state is entered
        _has_history: Whether this state remembers its last active substate
        _history_state: Last active substate (if history is enabled)

    Runtime Invariants:
        - Substates list is immutable after initialization
        - Initial state must be one of the substates if substates is not empty
        - History is only tracked if has_history is True
        - Last active state must be one of the substates
    """

    def __init__(
        self,
        state_id: StateID,
        substates: List[AbstractState],
        initial_state: Optional[AbstractState] = None,
        has_history: bool = False,
        parent_state: Optional[AbstractState] = None,
    ):
        """
        Initialize a new composite state.

        Args:
            state_id: Unique identifier for this state
            substates: List of states contained within this composite state
            initial_state: Default state to enter when this composite state is entered
            has_history: Whether to track the last active substate
            parent_state: Optional parent state for hierarchical state machines

        Raises:
            ValueError: If state_id is empty, initial_state is not in substates,
                      or if there are duplicate substates
            TypeError: If substates is not a list
        """
        # Call State's __init__ first
        super().__init__(state_id)

        if not isinstance(substates, list):
            raise ValueError("substates must be a list")

        # Check for duplicate substates
        seen_states = set()
        for state in substates:
            state_id = state.get_id()
            if state_id in seen_states:
                raise ValueError(f"Duplicate substate found with ID: {state_id}")
            seen_states.add(state_id)

        # Store an immutable copy of substates
        self._substates = tuple(substates)  # Make immutable at creation time
        self._current_substate: Optional[AbstractState] = None
        self._parent_state = parent_state

        # Initialize history-related attributes
        self._has_history = has_history
        self._history_state: Optional[AbstractState] = None

        # Validate and set initial state
        if substates:
            if initial_state is None:
                self._initial_state = substates[0]
            elif initial_state not in substates:
                raise ValueError("initial_state must be one of the substates")
            else:
                self._initial_state = initial_state
        else:
            if initial_state is not None:
                raise ValueError("initial_state must be None when substates is empty")
            self._initial_state = None

    @property
    def parent_state(self) -> Optional[AbstractState]:
        return self._parent_state

    @parent_state.setter
    def parent_state(self, value: AbstractState) -> None:
        self._parent_state = value

    def on_entry(self, event: AbstractEvent, data: Any) -> None:
        self._enter_substate(event, data)

    def on_exit(self, event: AbstractEvent, data: Any) -> None:
        if self._current_substate:
            if self.has_history():
                self.set_history_state(self._current_substate)
            self._current_substate.on_exit(event, data)

    def _enter_substate(self, event: AbstractEvent, data: Any) -> None:
        """Enter the appropriate substate."""
        if self.has_history() and self._history_state:
            self._current_substate = self._history_state
        else:
            self._current_substate = self.get_initial_state()

        if self._current_substate:
            self._current_substate.on_entry(event, data)

    def get_substates(self) -> List[AbstractState]:
        """
        Get the list of substates.

        Returns:
            A list of the substates
        """
        return list(self._substates)  # Return a new list copy

    def get_initial_state(self) -> AbstractState:
        """Get the initial substate."""
        if self._initial_state is None:
            raise ValueError("Initial state not set for composite state")
        return self._initial_state

    def has_history(self) -> bool:
        """Check if this state maintains history."""
        return self._has_history

    def set_history_state(self, state: AbstractState) -> None:
        """Update the history state."""
        if not self.has_history():
            raise ValueError("Cannot set history state when history is disabled")
        if state not in self._substates:
            raise ValueError("State is not a substate of this composite state")
        self._history_state = state
