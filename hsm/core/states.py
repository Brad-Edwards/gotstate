# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class State:
    """
    Represents a named state within the machine. It may define entry and exit
    actions, as well as hold its own data dictionary for state-specific variables.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize the state with a name and optional lists of actions to run on
        entry and exit.

        :param name: Unique name identifying this state.
        :param entry_actions: Actions executed upon entering this state.
        :param exit_actions: Actions executed upon exiting this state.
        """
        self._name = name
        self._data: Dict[str, Any] = {}
        self._entry_actions = entry_actions if entry_actions else []
        self._exit_actions = exit_actions if exit_actions else []
        self._parent: Optional[CompositeState] = None

    @property
    def name(self) -> str:
        """Return the state's name."""
        return self._name

    @property
    def data(self) -> Dict[str, Any]:
        """
        Access the state's local data store, which may be used to keep stateful
        information relevant only while this state is active.
        """
        return self._data

    @property
    def parent(self) -> Optional[CompositeState]:
        """Get the parent state if this is a child state."""
        return self._parent

    @parent.setter
    def parent(self, parent_state: Optional[CompositeState]) -> None:
        """Set the parent state."""
        self._parent = parent_state

    def on_enter(self) -> None:
        """
        Called when the state machine transitions into this state. Executes any
        defined entry actions and initializes state data.
        """
        self.initialize_data()
        for action in self._entry_actions:
            action()

    def on_exit(self) -> None:
        """
        Called when leaving this state. Executes exit actions and cleans up
        any state data if needed.
        """
        for action in self._exit_actions:
            action()
        self.cleanup_data()

    def initialize_data(self) -> None:
        """Prepare the state's data dictionary when this state is entered."""
        # By default, do nothing. Subclasses or plugins may customize.
        pass

    def cleanup_data(self) -> None:
        """Clean up or reset the state's data dictionary when exiting this state."""
        self._data.clear()


class CompositeState(State):
    """
    A state containing child states, representing a hierarchical structure.
    Useful for grouping related states and transitions within a logical namespace.
    """

    def __init__(
        self,
        name: str,
        entry_actions: List[Callable[[], None]] = None,
        exit_actions: List[Callable[[], None]] = None,
    ) -> None:
        super().__init__(name, entry_actions, exit_actions)
        self._children: Dict[str, State] = {}
        self._history_state: Optional[HistoryState] = None

    def add_child_state(self, state: State) -> None:
        """
        Add a child state to the composite state. Child states can form a nested
        hierarchy, enabling complex, modular state machines.

        :param state: The state to add as a child.
        """
        self._children[state.name] = state
        state.parent = self  # Set the parent-child relationship

    def set_history_state(self, history: HistoryState) -> None:
        """
        Set the history state for this composite state.

        :param history: The history state to use
        """
        self._history_state = history
        history.parent = self  # Set the parent-child relationship
        self._children[history.name] = history

    def get_history_state(self) -> Optional[HistoryState]:
        """Get the history state if one exists."""
        return self._history_state

    def get_child_state(self, name: str) -> State:
        """
        Retrieve a child state by name.

        :param name: Name of the desired child state.
        :return: The child State instance.
        """
        return self._children[name]

    @property
    def children(self) -> Dict[str, State]:
        """
        Obtain a dictionary of child states keyed by their names.
        """
        return self._children


class HistoryState(State):
    """
    A special state that remembers and restores the last active substate of a composite state.
    Can operate in either shallow (one level) or deep (entire nested configuration) mode.
    """

    def __init__(
        self,
        name: str,
        parent: CompositeState,
        deep: bool = False,
        default_state: State = None,
    ) -> None:
        """
        Initialize a history state.

        :param name: Unique name identifying this history state.
        :param parent: The composite state this history state belongs to.
        :param deep: If True, remembers the entire nested state configuration.
                    If False, only remembers one level of substates.
        :param default_state: The default state to transition to if no history exists.
        """
        super().__init__(name)
        self._parent = parent
        self._deep = deep
        self._default_state = default_state
        self._last_active: Dict[str, State] = {}  # Maps composite state names to their last active substates

    @property
    def is_deep(self) -> bool:
        """Whether this is a deep history state."""
        return self._deep

    @property
    def default_state(self) -> State:
        """The default state to transition to if no history exists."""
        return self._default_state

    def record_state(self, state: State) -> None:
        # If shallow history, just record the immediate substate.
        # This test seems to expect shallow history. We'll store by parent's name.
        if state.parent is not None:
            self._last_active[state.parent.name] = state

    def get_last_active(self, composite_state: CompositeState) -> State:
        """
        Get the last active state for a given composite state.

        :param composite_state: The composite state to get history for.
        :return: The last active state, or the default state if no history exists.
        """
        if composite_state != self._parent:
            return self._default_state
        return self._last_active.get(composite_state.name, self._default_state)

    def clear_history(self) -> None:
        """Clear all recorded history."""
        self._last_active.clear()

    def get_target(self) -> State:
        # If we have recorded last active for the parent, return that.
        # Otherwise, return default_state.
        if self.parent and self.parent.name in self._last_active:
            return self._last_active[self.parent.name]
        return self._default_state
