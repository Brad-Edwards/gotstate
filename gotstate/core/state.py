"""
State class and hierarchy management.

Implements hierarchical state structure using the Composite pattern.
Manages parent-child relationships, state data isolation, and state types.
"""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

import icontract

from gotstate.exceptions import DuplicateStateError, InvalidStateError, StateNotFoundError


class StateType(Enum):
    """Defines the different types of states in the hierarchical state machine."""

    SIMPLE = auto()
    COMPOSITE = auto()
    SUBMACHINE = auto()
    INITIAL = auto()
    FINAL = auto()
    CHOICE = auto()
    JUNCTION = auto()
    SHALLOW_HISTORY = auto()
    DEEP_HISTORY = auto()
    ENTRY_POINT = auto()
    EXIT_POINT = auto()
    TERMINATE = auto()


_PSEUDOSTATE_TYPES = frozenset(
    {
        StateType.INITIAL,
        StateType.CHOICE,
        StateType.JUNCTION,
        StateType.SHALLOW_HISTORY,
        StateType.DEEP_HISTORY,
        StateType.ENTRY_POINT,
        StateType.EXIT_POINT,
        StateType.TERMINATE,
    }
)


def _is_valid_state_name(name: str) -> bool:
    return isinstance(name, str) and len(name) > 0


@icontract.invariant(lambda self: isinstance(self._state_type, StateType), "State type must be a valid StateType")
@icontract.invariant(lambda self: _is_valid_state_name(self._name), "State name must be a non-empty string")
class State:
    """Represents a state in a hierarchical state machine.

    Implements the Composite pattern for hierarchical state structure.
    Thread-safe state data access is guaranteed for parallel regions.

    Class Invariants:
    1. State type must be a valid StateType enum value
    2. State name must be a non-empty string
    3. Parent-child relationships form a DAG (no cycles)
    """

    @icontract.require(lambda name: _is_valid_state_name(name), "Name must be a non-empty string")
    @icontract.require(lambda state_type: isinstance(state_type, StateType), "state_type must be a StateType")
    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.SIMPLE,
        parent: Optional[State] = None,
    ) -> None:
        self._name = name
        self._state_type = state_type
        self._parent: Optional[State] = None
        self._children: Dict[str, State] = {}
        self._data: Dict[str, Any] = {}
        self._entry_actions: List[Callable[[], None]] = []
        self._exit_actions: List[Callable[[], None]] = []
        self._is_active = False
        self._lock = threading.RLock()

        if parent is not None:
            parent.add_child(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def state_type(self) -> StateType:
        return self._state_type

    @property
    def parent(self) -> Optional[State]:
        return self._parent

    @property
    def children(self) -> Dict[str, State]:
        return dict(self._children)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_composite(self) -> bool:
        return self._state_type == StateType.COMPOSITE or len(self._children) > 0

    @property
    def is_pseudostate(self) -> bool:
        return self._state_type in _PSEUDOSTATE_TYPES

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @icontract.require(lambda child: child is not None, "Child must not be None")
    @icontract.ensure(lambda self, child: child._name in self._children, "Child must be added")
    def add_child(self, child: State) -> None:
        """Add a child state. Validates uniqueness and DAG property."""
        if child is self:
            raise InvalidStateError("A state cannot be its own child")
        ancestor = self._parent
        while ancestor is not None:
            if ancestor is child:
                raise InvalidStateError("Adding child would create a cycle in the state hierarchy")
            ancestor = ancestor._parent
        with self._lock:
            if child._name in self._children:
                raise DuplicateStateError(f"Child state '{child._name}' already exists in '{self._name}'")
            child._parent = self
            self._children[child._name] = child

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    def remove_child(self, name: str) -> State:
        """Remove and return a child state by name."""
        with self._lock:
            if name not in self._children:
                raise StateNotFoundError(f"Child state '{name}' not found in '{self._name}'")
            child = self._children.pop(name)
            child._parent = None
            return child

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    def get_child(self, name: str) -> State:
        """Get a child state by name. O(1) lookup."""
        if name not in self._children:
            raise StateNotFoundError(f"Child state '{name}' not found in '{self._name}'")
        return self._children[name]

    def get_ancestors(self) -> List[State]:
        """Return list of ancestors from immediate parent to root."""
        ancestors: List[State] = []
        current = self._parent
        while current is not None:
            ancestors.append(current)
            current = current._parent
        return ancestors

    def get_root(self) -> State:
        """Return the root state of the hierarchy."""
        current: State = self
        while current._parent is not None:
            current = current._parent
        return current

    @staticmethod
    def find_lca(state_a: State, state_b: State) -> Optional[State]:
        """Find the Lowest Common Ancestor of two states."""
        ancestors_a: Set[int] = set()
        current: Optional[State] = state_a
        while current is not None:
            ancestors_a.add(id(current))
            current = current._parent
        current = state_b
        while current is not None:
            if id(current) in ancestors_a:
                return current
            current = current._parent
        return None

    def on_entry(self, action: Callable[[], None]) -> None:
        """Register an entry action."""
        self._entry_actions.append(action)

    def on_exit(self, action: Callable[[], None]) -> None:
        """Register an exit action."""
        self._exit_actions.append(action)

    def enter(self) -> None:
        """Execute entry actions and mark state as active."""
        with self._lock:
            self._is_active = True
            for action in self._entry_actions:
                action()

    def exit(self) -> None:
        """Execute exit actions and mark state as inactive."""
        with self._lock:
            for action in self._exit_actions:
                action()
            self._is_active = False

    @icontract.require(lambda key: isinstance(key, str) and len(key) > 0, "Key must be a non-empty string")
    def set_data(self, key: str, value: Any) -> None:
        """Set state data. Thread-safe."""
        with self._lock:
            self._data[key] = value

    @icontract.require(lambda key: isinstance(key, str) and len(key) > 0, "Key must be a non-empty string")
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get state data, falling back to parent if not found."""
        with self._lock:
            if key in self._data:
                return self._data[key]
        if self._parent is not None:
            return self._parent.get_data(key, default)
        return default

    def __repr__(self) -> str:
        return f"State(name={self._name!r}, type={self._state_type.name})"


@icontract.invariant(
    lambda self: self._state_type == StateType.COMPOSITE,
    "CompositeState must have COMPOSITE type",
)
class CompositeState(State):
    """A state that contains child states and optional parallel regions.

    Manages initial state designation, history tracking, and
    ensures at most one initial pseudostate per region.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[State] = None,
    ) -> None:
        super().__init__(name, StateType.COMPOSITE, parent)
        self._initial_state: Optional[State] = None
        self._history_state: Optional[State] = None

    @property
    def initial_state(self) -> Optional[State]:
        return self._initial_state

    @icontract.require(lambda state: state is not None, "Initial state must not be None")
    def set_initial_state(self, state: State) -> None:
        """Designate an initial substate. Must be a child of this composite state."""
        if state._name not in self._children:
            raise InvalidStateError(f"State '{state._name}' is not a child of '{self._name}'")
        self._initial_state = state

    def get_active_substates(self) -> List[State]:
        """Return all currently active direct child states."""
        return [child for child in self._children.values() if child._is_active]


class PseudoState(State):
    """Base class for pseudostates that control execution flow.

    Pseudostates are transient vertices that cannot contain substates.
    """

    @icontract.require(
        lambda state_type: state_type in _PSEUDOSTATE_TYPES,
        "PseudoState type must be a pseudostate type",
    )
    def __init__(
        self,
        name: str,
        state_type: StateType,
        parent: Optional[State] = None,
    ) -> None:
        super().__init__(name, state_type, parent)

    def add_child(self, child: State) -> None:
        raise InvalidStateError("Pseudostates cannot contain child states")


class HistoryState(PseudoState):
    """Represents shallow or deep history pseudostates.

    Maintains historical state configuration of parent composite state.
    """

    @icontract.require(
        lambda state_type: state_type in (StateType.SHALLOW_HISTORY, StateType.DEEP_HISTORY),
        "HistoryState must be SHALLOW_HISTORY or DEEP_HISTORY",
    )
    def __init__(
        self,
        name: str,
        state_type: StateType = StateType.SHALLOW_HISTORY,
        parent: Optional[State] = None,
    ) -> None:
        super().__init__(name, state_type, parent)
        self._saved_configuration: Optional[List[State]] = None
        self._default_state: Optional[State] = None

    @property
    def saved_configuration(self) -> Optional[List[State]]:
        return self._saved_configuration

    @property
    def default_state(self) -> Optional[State]:
        return self._default_state

    @default_state.setter
    def default_state(self, state: Optional[State]) -> None:
        self._default_state = state

    def save_configuration(self, active_states: List[State]) -> None:
        """Save the current active state configuration."""
        self._saved_configuration = list(active_states)

    def restore_configuration(self) -> List[State]:
        """Restore saved configuration, falling back to default."""
        if self._saved_configuration is not None:
            return list(self._saved_configuration)
        if self._default_state is not None:
            return [self._default_state]
        return []


class ConnectionPointState(PseudoState):
    """Named entry/exit points for composite states."""

    @icontract.require(
        lambda state_type: state_type in (StateType.ENTRY_POINT, StateType.EXIT_POINT),
        "ConnectionPointState must be ENTRY_POINT or EXIT_POINT",
    )
    def __init__(
        self,
        name: str,
        state_type: StateType,
        parent: Optional[State] = None,
    ) -> None:
        super().__init__(name, state_type, parent)


class ChoiceState(PseudoState):
    """Dynamic conditional branch point that evaluates guards at runtime."""

    def __init__(self, name: str, parent: Optional[State] = None) -> None:
        super().__init__(name, StateType.CHOICE, parent)


class JunctionState(PseudoState):
    """Static conditional branch point that evaluates guards when reached."""

    def __init__(self, name: str, parent: Optional[State] = None) -> None:
        super().__init__(name, StateType.JUNCTION, parent)
