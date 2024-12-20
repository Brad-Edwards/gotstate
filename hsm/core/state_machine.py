# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from hsm.core.events import Event
from hsm.core.hooks import HookManager, HookProtocol
from hsm.core.runtime.context import RuntimeContext
from hsm.core.runtime.graph import StateGraph
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition, _TransitionPrioritySorter
from hsm.core.validations import ValidationError, Validator


@dataclass(frozen=True)
class _StateHistoryRecord:
    """Immutable record of historical state information"""

    timestamp: float
    state: State
    composite_state: CompositeState


class _StateMachineContext:
    """
    Internal context management for StateMachine, tracking current state and
    available transitions. Not for direct use by library clients.
    """

    def __init__(self, initial_state: State) -> None:
        self._current_state = initial_state
        self._transitions: List[Transition] = []
        self._states = {initial_state}  # Track all states
        self._history: Dict[CompositeState, _StateHistoryRecord] = {}
        self._history_lock = threading.Lock()

    def get_current_state(self) -> State:
        return self._current_state

    def set_current_state(self, state: State) -> None:
        self._current_state = state
        self._states.add(state)

    def get_transitions(self) -> List[Transition]:
        return self._transitions

    def get_states(self) -> set[State]:
        return self._states

    def add_transition(self, transition: Transition) -> None:
        self._transitions.append(transition)
        # Track states from transitions
        self._states.add(transition.source)
        self._states.add(transition.target)

    def start(self) -> None:
        """Start the context, initializing the current state."""
        if self._current_state:
            self._current_state.on_enter()

    def stop(self) -> None:
        """Stop the context, cleaning up the current state."""
        if self._current_state:
            self._current_state.on_exit()

    def process_event(self, event: Event) -> None:
        """Process an event in the current context."""
        # Implementation similar to StateMachine.process_event
        valid_transitions = [
            t for t in self._transitions if t.source == self._current_state and t.evaluate_guards(event)
        ]
        if valid_transitions:
            transition = sorted(valid_transitions, key=lambda t: t.get_priority(), reverse=True)[0]
            self._current_state.on_exit()
            transition.execute_actions(event)
            self._current_state = transition.target
            self._current_state.on_enter()

    def record_state_exit(self, composite_state: CompositeState, active_state: State) -> None:
        """Thread-safe recording of state history"""
        with self._history_lock:
            self._history[composite_state] = _StateHistoryRecord(
                timestamp=time.time(), state=active_state, composite_state=composite_state
            )

    def get_history_state(self, composite_state: CompositeState) -> Optional[State]:
        """Get the last active state for a composite state."""
        record = self._history.get(composite_state)
        return record.state if record else None


class _ErrorRecoveryStrategy:
    """
    Abstract interface for custom error recovery strategies. Provides a hook
    to handle exceptions within the state machine lifecycle.
    """

    def recover(self, error: Exception, state_machine: "StateMachine") -> None:
        # Default: do nothing. Subclasses can implement custom logic.
        pass


class StateMachine:
    """
    A finite state machine implementation that supports hierarchical states,
    prioritized transitions, and history states.
    """

    def __init__(self, initial_state: State, validator: Optional[Validator] = None, hooks: Optional[List] = None):
        """Initialize the state machine with an initial state."""
        self._graph = StateGraph()
        self._initial_state = initial_state
        self._current_state = initial_state  # Set current state immediately
        self._validator = validator or Validator()
        self._hooks = hooks or []
        self._started = False
        self._context = _StateMachineContext(initial_state)

        # Add initial state to graph
        self._graph.add_state(initial_state)

    def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """Add a state to the machine."""
        self._graph.add_state(state, parent)
        # Update internal state tracking
        if parent is not None:
            state.parent = parent
            if hasattr(parent, "_children"):
                parent._children.add(state)
        # Update context's state set directly
        self._context._states.add(state)

    @property
    def current_state(self) -> Optional[State]:
        """Get the current state."""
        return self._current_state

    def get_current_state(self) -> Optional[State]:
        """Get the current state (deprecated, use current_state property)."""
        return self._current_state

    def add_transition(self, transition: Transition) -> None:
        """Add a transition to the machine."""
        self._graph.add_transition(transition)
        self._context.add_transition(transition)

    def get_history_state(self, composite_state: CompositeState) -> Optional[State]:
        """Get the last active state for a composite state."""
        return self._context.get_history_state(composite_state)

    def start(self) -> None:
        """Start the state machine."""
        if self._started:
            return

        # Restore current state from initial if None
        if self._current_state is None:
            self._current_state = self._initial_state

        # Check for history state if we're in a composite state
        if (
            self._current_state.parent
            and isinstance(self._current_state.parent, CompositeState)
            and self._context._history
        ):  # Only check history if it exists
            history_state = self._context.get_history_state(self._current_state.parent)
            if history_state:
                self._current_state = history_state

        # Validate machine structure
        errors = self._graph.validate()
        if errors:
            self._current_state = None  # Reset state if validation fails
            raise ValidationError("\n".join(errors))

        self._validator.validate_state_machine(self)
        self._notify_enter(self._current_state)
        self._started = True

    def stop(self) -> None:
        """Stop the state machine."""
        if not self._started:
            return

        if self._current_state:
            # Record history for composite states before stopping
            if self._current_state.parent and isinstance(self._current_state.parent, CompositeState):
                self._context.record_state_exit(self._current_state.parent, self._current_state)

            self._notify_exit(self._current_state)
            self._current_state = None

        self._started = False

    def process_event(self, event: Event) -> bool:
        """Process an event and perform any valid transitions."""
        if not self._started or not self._current_state:
            return False

        # If current state is composite, use its active child state for transitions
        active_state = self._current_state
        if isinstance(active_state, CompositeState):
            active_state = active_state._initial_state

        valid_transitions = self._graph.get_valid_transitions(active_state, event)
        if not valid_transitions:
            return False

        # Take highest priority transition
        transition = max(valid_transitions, key=lambda t: t.get_priority())
        self._execute_transition(transition, event)
        return True

    def _execute_transition(self, transition: Transition, event: Event) -> None:
        """Execute a transition between states."""
        if not self._current_state:
            return

        try:
            # Record history for all ancestor composite states
            ancestors = self._graph.get_ancestors(self._current_state)
            for ancestor in ancestors:
                if isinstance(ancestor, CompositeState):
                    self._context.record_state_exit(ancestor, self._current_state)

            # Exit current state
            self._notify_exit(self._current_state)

            # Execute transition actions
            transition.execute_actions(event)

            # Enter new state
            self._current_state = transition.target
            self._notify_enter(self._current_state)

        except Exception as e:
            # Notify hooks of error
            self._notify_error(e)

            # If we have an error recovery strategy, use it
            if hasattr(self, "_error_recovery"):
                self._error_recovery.recover(e, self)
            else:
                # Re-raise if no recovery strategy
                raise

    def _notify_enter(self, state: State) -> None:
        """Notify hooks of state entry."""
        state.on_enter()
        for hook in self._hooks:
            if hasattr(hook, "on_enter"):
                hook.on_enter(state)

    def _notify_exit(self, state: State) -> None:
        """Notify hooks of state exit."""
        state.on_exit()
        for hook in self._hooks:
            if hasattr(hook, "on_exit"):
                hook.on_exit(state)

    def _notify_error(self, error: Exception) -> None:
        """Notify hooks of an error."""
        for hook in self._hooks:
            if hasattr(hook, "on_error"):
                hook.on_error(error)

    def detect_cycles(self) -> List[str]:
        """Detect cycles in the state hierarchy."""
        visited = set()
        path = set()
        cycles = []

        def visit(state):
            if state in path:
                cycle_path = [s.name for s in path]
                cycles.append(f"Cycle detected: {' -> '.join(cycle_path)}")
                return

            if state in visited:
                return

            visited.add(state)
            path.add(state)

            if isinstance(state, CompositeState):
                for child in state._children:
                    visit(child)

            path.remove(state)

        visit(self._initial_state)
        return cycles


class CompositeStateMachine(StateMachine):
    """
    A hierarchical extension of StateMachine that can contain nested submachines
    under composite states. This allows complex modeling of state hierarchies.
    """

    def __init__(self, initial_state: State, validator: Optional[Validator] = None, hooks: Optional[List] = None):
        super().__init__(initial_state, validator, hooks)
        self._submachines = {}

    def add_submachine(self, state: CompositeState, submachine: StateMachine) -> None:
        """Add a submachine for a composite state."""
        if not isinstance(state, CompositeState):
            raise ValueError(f"State {state.name} must be a composite state")

        # Register all states from submachine in our graph
        for sub_state in submachine._context.get_states():
            self._graph.add_state(sub_state, parent=state)

        self._submachines[state] = submachine

    def start(self) -> None:
        """Start the composite state machine and maintain composite state hierarchy."""
        super().start()
        # Don't automatically enter submachine states
        # Let the submachine handle its own state entry
        self._current_state = self._initial_state

    def process_event(self, event: Event) -> bool:
        """Process events in both the composite machine and relevant submachines."""
        # First try to process in current submachine if we're in a composite state
        if isinstance(self._current_state, CompositeState):
            submachine = self._submachines.get(self._current_state)
            if submachine and submachine.process_event(event):
                return True

        # If submachine didn't handle it, try processing at this level
        return super().process_event(event)
