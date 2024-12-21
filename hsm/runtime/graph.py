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
        self._transitions: Dict[State, Set[Transition]] = {}
        self._history: Dict[CompositeState, _StateHistoryRecord] = {}
        self._history_lock = threading.Lock()

    def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """Add a state to the graph with optional parent."""
        # Create new node if state doesn't exist
        if state not in self._nodes:
            self._nodes[state] = _GraphNode(state=state)
            self._transitions[state] = set()

        node = self._nodes[state]

        # Handle parent relationship
        if parent:
            if parent not in self._nodes:
                self.add_state(parent)

            parent_node = self._nodes[parent]

            # Check for cycles before updating relationships
            if self._would_create_cycle(state, parent):
                raise ValueError(f"Adding state {state.name} as child of {parent.name} would create a cycle")

            # Update parent-child relationships
            if node.parent:
                # Remove from old parent's children
                node.parent.children.remove(node)

            parent_node.children.add(node)
            node.parent = parent_node
            state.parent = parent

    def _would_create_cycle(self, state: State, new_parent: State) -> bool:
        """Check if adding state under new_parent would create a cycle."""
        current = self._nodes[new_parent]
        while current.parent:
            if current.parent.state == state:
                return True
            current = current.parent
        return False

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the graph."""
        if transition.source not in self._nodes:
            raise ValueError(f"Source state {transition.source.name} not in graph")
        if transition.target not in self._nodes:
            raise ValueError(f"Target state {transition.target.name} not in graph")

        self._transitions[transition.source].add(transition)
        self._nodes[transition.source].transitions.add(transition)

    def get_valid_transitions(self, state: State, event: Event) -> List[Transition]:
        """Get all valid transitions from a state for an event."""
        if state not in self._transitions:
            return []
        return sorted(
            [t for t in self._transitions[state] if t.evaluate_guards(event)],
            key=lambda t: t.get_priority(),
            reverse=True,
        )

    def get_ancestors(self, state: State) -> List[State]:
        """Get all ancestor states in order from immediate parent to root."""
        if state not in self._nodes:
            return []

        ancestors = []
        current = self._nodes[state]
        while current.parent:
            current = current.parent
            ancestors.append(current.state)
        return ancestors

    def get_children(self, state: State) -> Set[State]:
        """Get immediate child states of a state."""
        if state not in self._nodes:
            return set()
        return {node.state for node in self._nodes[state].children}

    def get_root_states(self) -> Set[State]:
        """Get all states that have no parent."""
        return {node.state for node in self._nodes.values() if not node.parent}

    def validate(self) -> List[str]:
        """Validate the graph structure."""
        errors = []
        visited = set()
        path = []

        def detect_cycle(state: State) -> None:
            if state in path:
                cycle_start = path.index(state)
                cycle_path = [s.name for s in path[cycle_start:]] + [state.name]
                errors.append(f"Cycle detected in state hierarchy: {' -> '.join(cycle_path)}")
                return

            if state in visited:
                return

            visited.add(state)
            path.append(state)

            node = self._nodes[state]
            for child_node in node.children:
                detect_cycle(child_node.state)

            path.pop()

        # Start cycle detection from root states
        root_states = [state for state in self._nodes.keys() if not self._nodes[state].parent]

        # If no root states found and graph is not empty, there must be a cycle
        if not root_states and self._nodes:
            errors.append("No root states found - graph contains cycles")
            # Start from any state to find the cycle
            detect_cycle(next(iter(self._nodes.keys())))
        else:
            for state in root_states:
                detect_cycle(state)

        # Validate composite states
        for node in self._nodes.values():
            state = node.state
            if isinstance(state, CompositeState):
                if not node.children:
                    errors.append(f"Composite state '{state.name}' has no children")
                if not state._initial_state and node.children:
                    first_child = next(iter(node.children)).state
                    state._initial_state = first_child

        return errors

    def record_history(self, composite_state: CompositeState, active_state: State) -> None:
        """Thread-safe history recording"""
        with self._history_lock:
            self._history[composite_state] = _StateHistoryRecord(
                timestamp=time.time(), state=active_state, composite_state=composite_state
            )

    def resolve_active_state(self, state: State) -> State:
        """Resolve the actual active state considering history and hierarchy"""
        if isinstance(state, CompositeState):
            # Check history first
            history_state = self._history.get(state)
            if history_state:
                return history_state.state
            # Fall back to initial state
            return state._initial_state or state
        return state

    def get_composite_ancestors(self, state: State) -> List[CompositeState]:
        """Get only composite state ancestors"""
        return [s for s in self.get_ancestors(state) if isinstance(s, CompositeState)]

    def clear_history(self) -> None:
        """Clear all history records"""
        with self._history_lock:
            self._history.clear()

    def set_initial_state(self, composite: CompositeState, state: State) -> None:
        """Set the initial state for a composite state."""
        if composite not in self._nodes:
            raise ValueError(f"Composite state {composite.name} not in graph")
        if state not in self._nodes:
            raise ValueError(f"State {state.name} not in graph")
        composite._initial_state = state
