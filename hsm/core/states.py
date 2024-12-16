# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from copy import deepcopy
from typing import Any, Dict, List, Optional

from hsm.interfaces.abc import AbstractCompositeState, AbstractState
from hsm.interfaces.types import StateID


class State(AbstractState):
    """
    Base class for state implementations.

    This class provides a foundation for implementing states in a hierarchical state machine.
    It manages state data and lifecycle events (enter/exit).

    Attributes:
        _state_id: Unique identifier for the state
        _data: Dictionary storing state-specific data

    Runtime Invariants:
        - State ID is immutable after initialization
        - State data is isolated between states
        - Enter/exit calls are paired
        - Data dictionary cannot be reassigned
    """

    def __init__(self, state_id: StateID):
        """
        Initialize a new state.

        Args:
            state_id: Unique identifier for this state

        Raises:
            ValueError: If state_id is empty or only whitespace
            TypeError: If state_id is None
        """
        if state_id is None:
            raise TypeError("state_id cannot be None")
        if not state_id or state_id.strip() == "":
            raise ValueError("state_id cannot be empty")

        self._state_id = state_id
        self._data: Dict[str, Any] = {}

    @property
    def data(self) -> Dict[str, Any]:
        """Access to the state's data dictionary."""
        return self._data

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item in the data dictionary with deep copy protection."""
        self._data[key] = deepcopy(value)

    def __getitem__(self, key: str) -> Any:
        """Get an item from the data dictionary."""
        return deepcopy(self._data[key])

    def on_enter(self) -> None:
        """
        Called when the state is entered.

        This method should be overridden by concrete implementations to define
        entry behavior.

        Raises:
            NotImplementedError: If not overridden by concrete class
        """
        raise NotImplementedError("on_enter must be implemented by concrete state classes")

    def on_exit(self) -> None:
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


class CompositeState(AbstractCompositeState, State):
    """
    A state that can contain other states, enabling hierarchical state machines.

    This class extends the base State class to support hierarchical state structures
    with substates and history tracking.

    Attributes:
        _substates: List of contained states
        _initial_state: Default state to enter when this composite state is entered
        _has_history: Whether this state remembers its last active substate
        _last_active: Last active substate (if history is enabled)

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
        initial_state: Optional[AbstractState],
        has_history: bool = False,
    ):
        """
        Initialize a new composite state.

        Args:
            state_id: Unique identifier for this state
            substates: List of states contained within this composite state
            initial_state: Default state to enter when this composite state is entered
            has_history: Whether to track the last active substate

        Raises:
            ValueError: If state_id is empty or initial_state is not in substates
            TypeError: If substates is not a list
        """
        # Call State's __init__ first
        State.__init__(self, state_id)

        # Initialize history-related attributes first
        self._has_history = has_history
        self._last_active = None

        if not isinstance(substates, list):
            raise ValueError("substates must be a list")

        # Store substates
        self._substates = substates
        self._initial_state = initial_state

        # Validate initial state based on whether we have substates
        if substates:
            if initial_state is None:
                raise ValueError("initial_state cannot be None when substates exist")
            if initial_state not in substates:
                raise ValueError("initial_state must be one of the substates")
        else:
            if initial_state is not None:
                raise ValueError("initial_state must be None when substates is empty")

    def get_substates(self) -> List[AbstractState]:
        """
        Get the list of substates.

        Returns:
            List of states contained within this composite state
        """
        return self._substates

    def get_initial_state(self) -> AbstractState:
        """
        Get the initial substate.

        If history is enabled and there is a last active state,
        returns that instead of the initial state.

        Returns:
            The state to enter when this composite state is entered
        """
        if self._has_history and self._last_active is not None:
            return self._last_active
        return self._initial_state

    def has_history(self) -> bool:
        """
        Check if this state maintains history.

        Returns:
            True if this state tracks its last active substate
        """
        return self._has_history

    def set_last_active(self, state: AbstractState) -> None:
        """
        Update the last active substate.

        This is called by the state machine when a substate becomes active.

        Args:
            state: The newly active substate

        Raises:
            ValueError: If the state is not a substate of this composite state
            TypeError: If state is None
        """
        if state is None:
            raise ValueError("state must be a substate of this composite state")
        if state not in self._substates:
            raise ValueError("state must be a substate of this composite state")
        self._last_active = state

    def on_enter(self) -> None:
        """
        Handle entry to this composite state.

        This implementation is empty as the state machine handles
        entering the appropriate substate.
        """
        pass

    def on_exit(self) -> None:
        """
        Handle exit from this composite state.

        This implementation is empty as the state machine handles
        exiting the current substate.
        """
        pass

    @property
    def data(self) -> Dict[str, Any]:
        """
        Access the state's data dictionary.

        Returns:
            Dict containing state-specific data with deep copy protection
        """
        # Return a new dict with deep copied values for nested structures
        result = {}
        for key, value in self._data.items():
            if isinstance(value, (dict, list)):
                result[key] = deepcopy(value)
            else:
                result[key] = value
        return result
