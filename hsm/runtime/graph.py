"""Graph-based state machine structure management."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..core.base import StateBase
from ..core.errors import ValidationError
from ..core.events import Event
from ..core.states import CompositeState, State
from ..core.transitions import Transition


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
    Manages the structural relationships between states in a state machine.
    Provides efficient access to transitions and hierarchy information.
    """

    def __init__(self) -> None:
        self._nodes: Dict[State, _GraphNode] = {}
        # Use a set for transitions so we can do .add(transition)
        self._transitions: Dict[State, Set[Transition]] = {}
        self._history: Dict[CompositeState, _StateHistoryRecord] = {}
        self._history_lock = threading.Lock()
        self._parent_map: Dict[State, Optional[State]] = {}

    def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """
        Add a state with an optional parent.
        If the state is already in the graph, raise an error if this call would
        introduce a different parent than before (i.e., re-parenting).
        """

        # If state is already in the graph, check for re-parenting
        if state in self._nodes:
            existing_parent = self._parent_map[state]
            # If there's no change, do nothing and return
            if existing_parent == parent:
                return  # Parent is the same, or both None
            # Otherwise, disallow re-parenting
            raise ValueError(
                f"Cannot re-parent state '{state.name}' from '{existing_parent.name if existing_parent else None}' "
                f"to '{parent.name if parent else None}'. Re-parenting is disallowed."
            )

        # If there's a parent, ensure it is in the graph
        if parent is not None:
            if parent not in self._nodes:
                raise ValueError(f"Parent state '{parent.name}' must be added to the graph first")
            # Check for cycle
            if self._would_create_cycle(state, parent):
                raise ValueError(f"Adding state '{state.name}' to parent '{parent.name}' would create a cycle")

        # If we get here, the state is not yet in the graph
        new_node = _GraphNode(state=state)
        self._nodes[state] = new_node
        self._parent_map[state] = parent

        # If there's a parent, link them in the _GraphNode structure
        if parent is not None:
            parent_node = self._nodes[parent]
            new_node.parent = parent_node
            parent_node.children.add(new_node)

            # If parent is a CompositeState, also update parent's _children
            if isinstance(parent, CompositeState):
                parent._children.add(state)
                state.parent = parent

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

    def get_children(self, state: State) -> Set[State]:
        """Get immediate child states of a state."""
        node = self._nodes.get(state)
        if not node:
            return set()
        # Return the underlying .state from each child node
        return {child_node.state for child_node in node.children}

    def get_root_states(self) -> Set[State]:
        """Get all states that have no parent."""
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

                # If it has children but no _initial_state, pick one
                if node.children and not st._initial_state:
                    first_child = next(iter(node.children)).state
                    st._initial_state = first_child

        return errors

    def record_history(self, composite_state: CompositeState, active_state: State) -> None:
        """Thread-safe history recording."""
        with self._history_lock:
            self._history[composite_state] = _StateHistoryRecord(
                timestamp=time.time(), state=active_state, composite_state=composite_state
            )

    def resolve_active_state(self, state: State) -> State:
        """Resolve active state using cached history."""
        if isinstance(state, CompositeState):
            with self._history_lock:
                record = self._history.get(state)
                if record:
                    return record.state
            return state._initial_state or state
        return state

    def get_composite_ancestors(self, state: State) -> List[CompositeState]:
        """Get composite state ancestors efficiently."""
        ancestors = []
        current = self._parent_map.get(state)
        while current:
            if isinstance(current, CompositeState):
                ancestors.append(current)
            current = self._parent_map.get(current)
        return ancestors

    def clear_history(self) -> None:
        """Clear all history records."""
        with self._history_lock:
            self._history.clear()

    def set_initial_state(self, composite: CompositeState, state: State) -> None:
        """Set the initial state for a composite state."""
        if composite not in self._nodes:
            raise ValueError(f"Composite state {composite.name} not in graph")
        if state not in self._nodes:
            raise ValueError(f"State {state.name} not in graph")
        composite._initial_state = state

    def get_all_states(self) -> Set[State]:
        """Get all states in the graph efficiently."""
        return set(self._nodes.keys())
