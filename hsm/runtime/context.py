"""
Runtime context management for state machines, relying on StateGraph for history.
"""

import threading
import time
from typing import Optional

from ..core.events import Event
from ..core.states import CompositeState, State
from .concurrency import with_lock
from .graph import StateGraph


class RuntimeContext:
    """
    Manages the runtime state of a state machine.
    Handles current state tracking and thread-safe transitions.
    History is stored in the StateGraph to avoid duplication.
    """

    def __init__(self, graph: StateGraph, initial_state: State) -> None:
        self._graph = graph
        self._current_state = initial_state
        self._transition_lock = threading.Lock()

    def get_current_state(self) -> State:
        """Get the currently active state."""
        return self._current_state

    def process_event(self, event: Event) -> bool:
        with with_lock(self._transition_lock):
            valid_transitions = self._graph.get_valid_transitions(self._current_state, event)
            if not valid_transitions:
                return False

            transition = max(valid_transitions, key=lambda t: t.get_priority())

            self._current_state.on_exit()
            transition.execute_actions(event)
            self._current_state = transition.target
            self._current_state.on_enter()

            return True

    def _record_history_in_graph(self, state: State) -> None:
        """Use the graph to store history for each composite ancestor."""
        # For each composite ancestor, call graph.record_history()
        ancestors = self._graph.get_composite_ancestors(state)
        for ancestor in ancestors:
            self._graph.record_history(ancestor, state)

    def get_history_state(self, composite_state: CompositeState) -> Optional[State]:
        """
        Get the last active state for a composite state from the graph.
        """
        return self._graph.get_history_state(composite_state)
