# hsm/runtime/graph.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

"""Graph-based state machine structure management."""

import threading
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from ..core.base import StateBase
from ..core.errors import ValidationError
from ..core.events import Event
from ..core.states import CompositeState, State
from ..core.transitions import Transition
from ..runtime.state_history import StateHistory


@dataclass
class _StateHistoryRecord:
    """Internal record for state history tracking."""

    timestamp: float
    state: State
    composite_state: CompositeState


@dataclass
class _GraphNode:
    """Internal node representation for the state graph."""

    state: State
    transitions: Set[Transition] = field(default_factory=set)
    children: Set["_GraphNode"] = field(default_factory=set)
    parent: Optional["_GraphNode"] = None

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        if not isinstance(other, _GraphNode):
            return NotImplemented
        return self.state == other.state


class StateGraph:
    """
    Manages the structural relationships and data for all states in a state machine.
    Provides efficient access to transitions, hierarchy information, and state data.
    This is the single source of truth for all state-related information.
    """

    def __init__(self) -> None:
        self._nodes: Dict[State, _GraphNode] = {}
        self._transitions: Dict[State, Set[Transition]] = {}
        self._history = StateHistory()
        self._parent_map: Dict[State, Optional[State]] = {}
        self._initial_states: Dict[CompositeState, State] = {}
        self._state_data: Dict[State, Dict[str, Any]] = {}
        self._data_locks: Dict[State, threading.Lock] = {}
        self._graph_lock = threading.Lock()  # Lock for structural modifications
        self._resolution_lock = Lock()
        self._current_state: Optional[State] = None  # Track current active state

    def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """Add a state with an optional parent."""
        with self._graph_lock:
            # Initialize state data and lock
            if state not in self._state_data:
                self._state_data[state] = {}
                self._data_locks[state] = threading.Lock()

            # If state is already in the graph, check for re-parenting
            if state in self._nodes:
                existing_parent = self._parent_map[state]
                if existing_parent == parent:
                    return
                raise ValueError(
                    f"Cannot re-parent state '{state.name}' from '{existing_parent.name if existing_parent else None}' "
                    f"to '{parent.name if parent else None}'. Re-parenting is disallowed."
                )

            # Create and link the new node
            new_node = _GraphNode(state=state)
            self._nodes[state] = new_node
            self._parent_map[state] = parent

            if parent is not None:
                parent_node = self._nodes[parent]
                new_node.parent = parent_node
                parent_node.children.add(new_node)

    def get_state_data(self, state: State) -> Dict[str, Any]:
        """Thread-safe access to state data."""
        if state not in self._state_data:
            self._state_data[state] = state.data  # Use existing state data
            self._data_locks[state] = threading.Lock()
        return self._state_data[state]

    def set_state_data(self, state: State, key: str, value: Any) -> None:
        """Thread-safe way to set state data."""
        with self._data_locks[state]:
            if state not in self._state_data:
                self._state_data[state] = {}
            self._state_data[state][key] = value

    def get_initial_state(self, composite_state: CompositeState) -> Optional[State]:
        """Get the initial state for a composite state."""
        return self._initial_states.get(composite_state)

    def set_initial_state(self, composite_state: Optional[CompositeState], initial_state: State) -> None:
        """Set the initial state for a composite state or the root state machine.

        Args:
            composite_state: The composite state to set initial state for, or None for root state machine
            initial_state: The state to set as initial
        """
        if initial_state not in self._nodes:
            raise ValueError(f"Initial state '{initial_state.name}' not in graph")

        if composite_state is None:
            # Special case: setting root initial state
            self._initial_states[composite_state] = initial_state
            return

        if composite_state not in self._nodes:
            raise ValueError(f"Composite state '{composite_state.name}' not in graph")

        if initial_state not in self.get_children(composite_state):
            raise ValueError(f"Initial state '{initial_state.name}' must be a child of '{composite_state.name}'")

        self._initial_states[composite_state] = initial_state

    def get_parent(self, state: State) -> Optional[State]:
        """Get the parent state of a state."""
        return self._parent_map.get(state)

    def get_children(self, state: State) -> Set[State]:
        """Get the child states of a state."""
        if state not in self._nodes:
            return set()
        return {node.state for node in self._nodes[state].children}

    def is_composite_parent(self, state: State) -> bool:
        """Check if a state is parented by a CompositeState."""
        parent = self._parent_map.get(state)
        return isinstance(parent, CompositeState) if parent else False

    def _would_create_cycle(self, state: State, new_parent: State) -> bool:
        """Check if adding state under new_parent would create a cycle."""
        # First check if state is already in new_parent's ancestors
        current = new_parent
        while current:
            if current == state:
                return True
            current = self._parent_map.get(current)
        return False

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the graph."""
        with self._graph_lock:
            if transition.source not in self._nodes:
                raise ValueError(f"Source state {transition.source.name} not in graph")
            if transition.target not in self._nodes:
                raise ValueError(f"Target state {transition.target.name} not in graph")

            # Initialize if needed
            if transition.source not in self._transitions:
                self._transitions[transition.source] = set()

            self._transitions[transition.source].add(transition)
            self._nodes[transition.source].transitions.add(transition)

    def get_valid_transitions(self, state: State, event: Event) -> List[Transition]:
        """Get all transitions from a state for an event."""
        with self._graph_lock:
            if state not in self._transitions:
                return []
            # Return all transitions sorted by priority, let the state machine evaluate guards
            return sorted(list(self._transitions[state]), key=lambda t: t.get_priority(), reverse=True)

    def get_ancestors(self, state: State) -> List[State]:
        """Get all ancestor states in order from immediate parent to root."""
        if state not in self._nodes:
            return []

        ancestors = []
        current = self._parent_map.get(state)
        while current:
            ancestors.append(current)
            current = self._parent_map.get(current)
        return ancestors

    def get_root_states(self) -> Set[State]:
        """Get all states that have no parent."""
        with self._graph_lock:
            return {node.state for node in self._nodes.values() if node.parent is None}

    def validate(self) -> List[str]:
        """Validate the graph structure."""
        errors = []
        visited = set()
        path = []

        def detect_cycle(st: State) -> None:
            if st in path:
                cycle_start = path.index(st)
                cycle_path = [s.name for s in path[cycle_start:]] + [st.name]
                errors.append(f"Cycle detected in state hierarchy: {' -> '.join(cycle_path)}")
                return

            if st in visited:
                return

            visited.add(st)
            path.append(st)

            # Recurse on children
            for child_node in self._nodes[st].children:
                detect_cycle(child_node.state)

            path.pop()

        # Start cycle detection from root states
        for root_node in [n for n in self._nodes.values() if n.parent is None]:
            detect_cycle(root_node.state)

        # Validate composite states
        for node in self._nodes.values():
            st = node.state
            if isinstance(st, CompositeState):
                # If it's a composite with no children, that's possibly an error
                if not node.children:
                    errors.append(f"Composite state '{st.name}' has no children")

                # If it has children but no initial state, pick one
                if node.children and st not in self._initial_states:
                    errors.append(f"Composite state '{st.name}' has no initial state set")

        return errors

    def record_history(self, composite_state: CompositeState, active_state: State) -> None:
        """Thread-safe history recording."""
        self._history.record_state(composite_state, active_state)

    def resolve_active_state(self, state: State) -> State:
        """Thread-safe state resolution with granular locking."""
        if not isinstance(state, CompositeState):
            return state

        with self._resolution_lock:
            hist_state = self._history.get_last_state(state)
            if hist_state:
                # Release lock before recursive call
                next_state = hist_state
            else:
                initial = self.get_initial_state(state)
                if initial is None:
                    children = self.get_children(state)
                    if children:
                        initial = next(iter(children))
                        self.set_initial_state(state, initial)
                next_state = initial if initial else state

        # Do recursive resolution outside the lock
        if next_state != state:
            return self.resolve_active_state(next_state)
        return state

    def get_composite_ancestors(self, state: State) -> List[CompositeState]:
        """Get composite state ancestors efficiently."""
        ancestors = []
        current = self._parent_map.get(state)
        while current:
            if isinstance(current, CompositeState):
                ancestors.append(current)
            current = self._parent_map.get(current)
        # Return in outermost to innermost order
        return list(reversed(ancestors))

    def clear_history(self) -> None:
        """Clear all history records."""
        self._history.clear()

    def get_all_states(self) -> Set[State]:
        """Get all states in the graph efficiently."""
        with self._graph_lock:
            return set(self._nodes.keys())

    def get_history_state(self, composite: CompositeState) -> Optional[State]:
        """Get the last active state for a composite state."""
        return self._history.get_last_state(composite)

    def merge_submachine(self, parent: CompositeState, submachine_graph: "StateGraph") -> None:
        """
        Merge a submachine's graph into this graph under the given parent state.
        Preserves the initial state and transition relationships from the submachine.
        """
        # Use a shorter timeout for the lock to prevent hanging
        if not self._graph_lock.acquire(timeout=2.0):
            raise RuntimeError("Failed to acquire graph lock for merge operation")

        try:
            # First add all states from submachine to ensure they exist in our graph
            states_map = {}  # Map from submachine states to our graph states

            # Handle the parent state first since it's already in our graph
            states_map[parent] = parent

            # Then handle all other states
            for state in submachine_graph.get_all_states():
                if state == parent:
                    continue

                # Create a new state instance to avoid sharing state between graphs
                new_state = State(state.name) if isinstance(state, State) else CompositeState(state.name)

                # Initialize state data and lock
                if new_state not in self._state_data:
                    self._state_data[new_state] = {}
                    self._data_locks[new_state] = threading.Lock()

                # Create and link the new node directly (without re-acquiring lock)
                new_node = _GraphNode(state=new_state)
                self._nodes[new_state] = new_node
                self._parent_map[new_state] = parent

                if parent is not None:
                    parent_node = self._nodes[parent]
                    new_node.parent = parent_node
                    parent_node.children.add(new_node)

                # Copy state data using graph methods
                state_data = submachine_graph.get_state_data(state)
                if state_data:
                    for key, value in state_data.items():
                        with self._data_locks[new_state]:
                            self._state_data[new_state][key] = value

                states_map[state] = new_state

            # Add all transitions from submachine, updating state references
            for source_state in submachine_graph._transitions:
                for transition in submachine_graph._transitions[source_state]:
                    # Skip transitions that involve the parent state
                    if transition.source == parent or transition.target == parent:
                        continue

                    # Create a new transition with mapped states
                    new_transition = Transition(
                        source=states_map[transition.source],
                        target=states_map[transition.target],
                        guards=transition.guards,
                    )

                    # Add transition directly (without re-acquiring lock)
                    source = states_map[transition.source]
                    if source not in self._transitions:
                        self._transitions[source] = set()
                    self._transitions[source].add(new_transition)
                    self._nodes[source].transitions.add(new_transition)

            # Preserve initial state relationships from submachine
            submachine_initial = submachine_graph.get_initial_state(None)  # Get root initial state
            if submachine_initial and submachine_initial in states_map:
                self.set_initial_state(parent, states_map[submachine_initial])

        finally:
            self._graph_lock.release()

    def get_current_state(self) -> Optional[State]:
        """Get the current active state."""
        return self._current_state

    def set_current_state(self, state: Optional[State]) -> None:
        """Set the current active state."""
        if state is not None and state not in self._nodes:
            raise ValueError(f"State '{state.name}' not in graph")
        self._current_state = state
