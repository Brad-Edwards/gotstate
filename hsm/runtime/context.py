"""Runtime context management for state machines."""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set

from ..core.events import Event
from ..core.states import CompositeState, State
from .graph import StateGraph


@dataclass
class _StateHistoryRecord:
    """Immutable record of historical state information."""

    timestamp: float
    state: State
    composite_state: CompositeState


class RuntimeContext:
    """
    Manages the runtime state of a state machine.
    Handles current state tracking, history, and thread-safe transitions.
    """

    def __init__(self, graph: StateGraph, initial_state: State) -> None:
        self._graph = graph
        self._current_state = initial_state
        self._history: Dict[CompositeState, _StateHistoryRecord] = {}
        self._history_lock = threading.Lock()
        self._transition_lock = threading.Lock()

    def get_current_state(self) -> State:
        """Get the currently active state."""
        return self._current_state

    def process_event(self, event: Event) -> bool:
        """
        Process an event in the current context.
        Returns True if a transition was taken.
        """
        with self._transition_lock:
            # Get valid transitions from the graph
            valid_transitions = self._graph.get_valid_transitions(self._current_state, event)
            if not valid_transitions:
                return False

            # Take highest priority transition
            transition = max(valid_transitions, key=lambda t: t.get_priority())

            # Record history before exit
            self._record_history(self._current_state)

            # Execute transition
            self._current_state.on_exit()
            transition.execute_actions(event)
            self._current_state = transition.target
            self._current_state.on_enter()

            return True

    def _record_history(self, state: State) -> None:
        """Record state history for composite states."""
        with self._history_lock:
            # Find all ancestor composite states
            ancestors = self._graph.get_ancestors(state)
            for ancestor in ancestors:
                if isinstance(ancestor, CompositeState):
                    self._history[ancestor] = _StateHistoryRecord(
                        timestamp=time.time(), state=state, composite_state=ancestor
                    )

    def get_history_state(self, composite_state: CompositeState) -> Optional[State]:
        """Get the last active state for a composite state."""
        record = self._history.get(composite_state)
        return record.state if record else None
