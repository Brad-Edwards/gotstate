# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Any, Callable, Dict, List


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

    def add_child_state(self, state: State) -> None:
        """
        Add a child state to the composite state. Child states can form a nested
        hierarchy, enabling complex, modular state machines.

        :param state: The state to add as a child.
        """
        self._children[state.name] = state

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
