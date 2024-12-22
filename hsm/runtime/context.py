"""
Runtime context management for state machines, relying on StateGraph for history.
"""

import threading
import time
from typing import Optional

from hsm.core.transitions import Transition

from ..core.events import Event
from ..core.states import CompositeState, State
from .concurrency import with_lock
from .graph import StateGraph


class RuntimeContext:
    """
    Execution context for a state machine, managing state transitions
    and event processing. Handles composite states and initial state resolution.
    """

    def __init__(self, graph: StateGraph, initial_state: State):
        self._graph = graph
        self._initial_state = initial_state
        self._current_state = None
        # Initialize with the resolved initial state
        self._set_current_state(initial_state)

    def get_current_state(self) -> State:
        """Get the current state."""
        return self._current_state

    def _set_current_state(self, state: Optional[State]) -> None:
        """Internal method to update current state."""
        if state is None:
            self._current_state = None
            return
        # Resolve the state if it's a composite state
        resolved_state = self._graph.resolve_active_state(state)
        self._current_state = resolved_state

    def process_event(self, event: Event) -> bool:
        """
        Process an event, checking transitions from the current state.
        Execute the highest-priority valid transition, if any.
        """
        if self._current_state is None:
            return False

        valid_transitions = self._graph.get_valid_transitions(self._current_state, event)
        if not valid_transitions:
            return False

        # Pick the highest-priority transition
        transition = max(valid_transitions, key=lambda t: t.get_priority())
        self._execute_transition(transition, event)
        return True

    def _execute_transition(self, transition: Transition, event: Event) -> None:
        """Execute a transition, including exit/enter notifications."""
        # Notify exit
        self._current_state.on_exit()

        # Execute transition actions
        for action in transition.actions:
            action(event)

        # Update current state
        self._set_current_state(transition.target)

        # Notify enter
        self._current_state.on_enter()
