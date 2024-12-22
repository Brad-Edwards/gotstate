# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import time
from typing import Callable, List, Optional, Set

from hsm.core.events import Event
from hsm.core.hooks import HookManager, HookProtocol
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import ValidationError, Validator
from hsm.runtime.graph import StateGraph


class _ErrorRecoveryStrategy:
    """
    Abstract interface for custom error recovery strategies.
    Subclasses can implement custom logic in `recover`.
    """

    def recover(self, error: Exception, state_machine: "StateMachine") -> None:
        pass


class StateMachine:
    """
    A finite state machine implementation that uses StateGraph as the
    sole source of truth for hierarchy, transitions, and history.
    """

    def __init__(
        self,
        initial_state: State,
        validator: Optional[Validator] = None,
        hooks: Optional[List] = None,
        error_recovery: Optional[_ErrorRecoveryStrategy] = None,
    ):
        """
        :param initial_state: The state in which this machine begins.
        :param validator: Optional validator for structure checks.
        :param hooks: Optional list of hook objects implementing on_enter, on_exit, on_error, etc.
        :param error_recovery: Optional error recovery strategy.
        """
        self._graph = StateGraph()
        self._validator = validator or Validator()
        self._hooks = hooks or []
        self._error_recovery = error_recovery
        self._started = False

        # Add the initial state into the graph
        self._graph.add_state(initial_state)
        self._initial_state = initial_state
        self._current_state: Optional[State] = None
        self._set_current_state(initial_state)

        # If initial_state has a parent composite that doesn't have an initial, set it
        if isinstance(initial_state.parent, CompositeState):
            if not initial_state.parent._initial_state:
                initial_state.parent._initial_state = initial_state

    @property
    def current_state(self) -> Optional[State]:
        """Get the current active state."""
        return self._current_state

    def _set_current_state(self, state: Optional[State], notify: bool = False) -> None:
        """
        Internal method to update current state.
        :param state: The new state to set
        :param notify: Whether to notify hooks about state change
        """
        if notify and self._current_state:
            self._notify_exit(self._current_state)

        self._current_state = state

        if notify and state:
            self._notify_enter(state)

    def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """
        Add a state to the underlying graph, optionally specifying a parent.

        :param state: The state to add.
        :param parent: The parent state (usually a CompositeState).
        """
        self._graph.add_state(state, parent)
        # If parent is a CompositeState with no initial_state, set this new state
        if isinstance(parent, CompositeState) and not parent._initial_state:
            parent._initial_state = state

    def add_transition(self, transition: Transition) -> None:
        """
        Add a transition to the graph.
        The graph enforces that source/target states exist.
        """
        self._graph.add_transition(transition)

    def get_transitions(self) -> Set[Transition]:
        """
        Return all transitions from the graph.
        (In a minimal version, you'd store transitions in the StateGraph.)
        """
        # For example, we might collect them from all source states:
        transitions = set()
        for st in self._graph.get_all_states():
            for tr in self._graph._transitions.get(st, set()):
                transitions.add(tr)
        return transitions

    def get_states(self) -> Set[State]:
        """Get all states from the graph."""
        return self._graph.get_all_states()

    def get_history_state(self, composite: CompositeState) -> Optional[State]:
        """Retrieve a recorded history state from the graph."""
        return self._graph.get_history_state(composite)

    def start(self) -> None:
        """Start the state machine."""
        if self._started:
            return

        # Validate the graph structure
        errors = self._graph.validate()
        if errors:
            raise ValidationError("\n".join(errors))

        self._validator.validate_state_machine(self)

        # Resolve initial or historical active state
        resolved_state = self._graph.resolve_active_state(self._initial_state)
        
        # Set current state without notifications first
        self._set_current_state(resolved_state, notify=False)
        # Then only notify enter since we're starting up
        self._notify_enter(self._current_state)
        
        self._started = True

    def stop(self) -> None:
        """Stop the state machine."""
        if not self._started:
            return

        if self._current_state:
            # Record history if applicable
            composite_ancestors = self._graph.get_composite_ancestors(self._current_state)
            if composite_ancestors:
                self._graph.record_history(composite_ancestors[0], self._current_state)

            self._set_current_state(None, notify=True)

        self._started = False

    def process_event(self, event: Event) -> bool:
        """
        Process an event, checking transitions from the current state via the graph.
        Execute the highest-priority valid transition, if any.
        """
        if not self._started or not self._current_state:
            return False

        try:
            valid_transitions = self._graph.get_valid_transitions(self._current_state, event)
            if not valid_transitions:
                return False

            # Pick the highest-priority transition
            transition = valid_transitions[0]
            self._execute_transition(transition, event)
            return True

        except Exception as error:
            if self._error_recovery:
                self._error_recovery.recover(error, self)
            else:
                raise
            return False

    def _execute_transition(self, transition: Transition, event: Event) -> None:
        """Execute a transition, notify exit/enter, handle errors."""
        try:
            # First notify exit of current state
            if self._current_state:
                self._notify_exit(self._current_state)

            # Execute transition actions
            for action in transition.actions:
                action(event)

            # Update current state without notifications (they're handled here)
            self._set_current_state(transition.target, notify=False)

            # Notify enter of new state
            self._notify_enter(self._current_state)

        except Exception as e:
            self._notify_error(e)
            raise

    def _notify_enter(self, state: State) -> None:
        """Invoke on_enter hooks."""
        state.on_enter()
        for hook in self._hooks:
            if hasattr(hook, "on_enter") and not asyncio.iscoroutinefunction(hook.on_enter):
                hook.on_enter(state)

    def _notify_exit(self, state: State) -> None:
        """Invoke on_exit hooks."""
        state.on_exit()
        for hook in self._hooks:
            if hasattr(hook, "on_exit"):
                hook.on_exit(state)

    def _notify_error(self, error: Exception) -> None:
        """Invoke on_error hooks."""
        for hook in self._hooks:
            if hasattr(hook, "on_error"):
                hook.on_error(error)

    def reset(self) -> None:
        """Reset the state machine to its initial state (clearing all history)."""
        was_started = self._started
        if was_started:
            self.stop()
        self._graph.clear_history()
        # Restore current state to initial state before restarting
        self._set_current_state(self._initial_state)
        if was_started:
            self.start()

    def validate(self) -> List[str]:
        """Expose the graph's validation results."""
        return self._graph.validate()


class CompositeStateMachine(StateMachine):
    """
    A hierarchical extension of StateMachine that can contain nested submachines.
    Each submachine can be integrated into the same graph or kept separate.
    """

    def __init__(
        self,
        initial_state: State,
        validator: Optional[Validator] = None,
        hooks: Optional[List] = None,
        error_recovery: Optional[_ErrorRecoveryStrategy] = None,
    ):
        super().__init__(initial_state, validator, hooks, error_recovery)
        self._submachines = {}

    def add_submachine(self, state: CompositeState, submachine: "StateMachine") -> None:
        """
        Add a submachine's states under a parent composite state.
        Submachine's states are all integrated into this machine's graph.
        """
        if not isinstance(state, CompositeState):
            raise ValueError(f"State {state.name} must be a composite state")

        # Integrate submachine states into the same graph:
        for sub_state in submachine.get_states():
            # We assume sub_state's parent can be updated to "state"
            self._graph.add_state(sub_state, parent=state)

        # Integrate transitions
        for t in submachine.get_transitions():
            self._graph.add_transition(t)

        self._submachines[state] = submachine

    def process_event(self, event: Event) -> bool:
        """
        Try to process event in the current submachine if the current state is composite.
        Otherwise, process normally in this machine.
        """
        # First try to process in the current submachine if we're in a composite state
        if isinstance(self.current_state, CompositeState):
            submachine = self._submachines.get(self.current_state)
            if submachine and submachine.process_event(event):
                return True

        # If we're in a submachine state, check for transitions in the main machine's graph
        if self.current_state and any(
            isinstance(s, CompositeState) for s in self._graph.get_composite_ancestors(self.current_state)
        ):
            valid_transitions = self._graph.get_valid_transitions(self.current_state, event)
            if valid_transitions:
                transition = valid_transitions[0]  # Get highest priority transition
                self._execute_transition(transition, event)
                return True

        # If no submachine handled it and no transition in main machine, try normal processing
        return super().process_event(event)
