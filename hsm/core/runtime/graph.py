"""Graph-based state machine structure management."""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from ..base import StateBase
from ..states import State, CompositeState
from ..transitions import Transition
from ..events import Event
from ..errors import ValidationError


@dataclass
class _GraphNode:
    """Internal node representation for the state graph."""
    state: State
    transitions: Set[Transition] = field(default_factory=set)
    children: Set['_GraphNode'] = field(default_factory=set)
    parent: Optional['_GraphNode'] = None

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

    def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """Add a state to the graph with optional parent."""
        if state not in self._nodes:
            self._nodes[state] = _GraphNode(state=state)
            self._transitions[state] = set()

        if parent:
            if parent not in self._nodes:
                self.add_state(parent)
            
            parent_node = self._nodes[parent]
            state_node = self._nodes[state]
            
            # Update parent-child relationships
            parent_node.children.add(state_node)
            state_node.parent = parent_node
            state.parent = parent

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
            reverse=True
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
                cycle_names = [s.name for s in path[path.index(state):]] + [state.name]
                errors.append(f"Cycle detected in state hierarchy: {' -> '.join(cycle_names)}")
                return
            
            if state in visited:
                return
                
            visited.add(state)
            path.append(state)
            
            node = self._nodes[state]
            for child_node in node.children:
                detect_cycle(child_node.state)
                
            path.pop()

        # Check composite states have children
        for node in self._nodes.values():
            if isinstance(node.state, CompositeState) and not node.children:
                errors.append(f"Composite state '{node.state.name}' has no children")

        # Detect cycles starting from all states to catch disconnected cycles
        for state in self._nodes:
            if state not in visited:
                detect_cycle(state)

        return errors