# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
from typing import List, Optional, Set
from threading import Lock

from hsm.core.events import Event
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
    A hierarchical state machine implementation that manages states, transitions, and history.

    Features:
    - Hierarchical state nesting
    - History state tracking
    - Event-driven transitions
    - Guard conditions
    - Entry/exit actions
    - Custom hooks for state changes
    - Error recovery strategies
    """

    def __init__(
        self,
        initial_state: State,
        validator: Optional[Validator] = None,
        hooks: Optional[List] = None,
        error_recovery: Optional[_ErrorRecoveryStrategy] = None,
    ):
        """
        Initialize a new state machine.

        Args:
            initial_state: The starting state of the machine
            validator: Optional validator to perform structural validation checks
            hooks: Optional list of hook objects that can respond to state changes.
                  Hooks can implement: on_enter(state), on_exit(state),
                  on_transition(source, target), and on_error(error)
            error_recovery: Optional strategy for handling runtime errors
        """
        self._graph = StateGraph()
        self._validator = validator or Validator()
        self._hooks = hooks or []
        self._error_recovery = error_recovery
        self._started = False
        self._transition_lock = Lock()

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
        Add a state to the state machine.

        Args:
            state: The state to add
            parent: Optional parent state for hierarchical nesting. If parent is a
                   CompositeState with no initial state set, this state becomes its
                   initial state.
        """
        self._graph.add_state(state, parent)
        # If parent is a CompositeState with no initial_state, set this new state
        if isinstance(parent, CompositeState) and not parent._initial_state:
            parent._initial_state = state

    def add_transition(self, transition: Transition) -> None:
        """
        Add a transition between states.

        Args:
            transition: The transition to add. Must reference states that exist
                      in the state machine.
        """
        self._graph.add_transition(transition)

    def get_transitions(self) -> Set[Transition]:
        """
        Return all transitions from the graph.
        """
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
        """
        Start the state machine.

        This validates the machine's structure and enters the initial state,
        triggering any relevant entry actions and hooks.

        Raises:
            ValidationError: If the state machine structure is invalid
        """
        with self._transition_lock:
            if self._started:
                return

            # Validate before starting
            errors = self._graph.validate()
            if errors:
                raise ValidationError("\n".join(errors))

            self._validator.validate_state_machine(self)
            self._validate_composite_states()

            # Resolve initial or historical active state
            resolved_state = self._graph.resolve_active_state(self._initial_state)

            # Set current state without notifications first
            self._set_current_state(resolved_state, notify=False)

            # Get all ancestors of the current state and notify enter from outermost to innermost
            ancestors = self._graph.get_composite_ancestors(self._current_state)
            # Enter ancestors first (they're already in outermost to innermost order)
            for ancestor in ancestors:
                self._notify_enter(ancestor)
            self._notify_enter(self._current_state)

            self._started = True

    def stop(self) -> None:
        """
        Stop the state machine.

        This exits all active states (triggering exit actions), records history
        states, and resets the current state to None.
        """
        if not self._started:
            return

        if self._current_state:
            composite_ancestors = self._graph.get_composite_ancestors(self._current_state)
            if composite_ancestors:
                for ancestor in composite_ancestors:
                    self._graph.record_history(ancestor, self._current_state)

            # Exit from innermost to outermost
            current = self._current_state
            self._notify_exit(current)
            for ancestor in reversed(composite_ancestors):
                self._notify_exit(ancestor)

            self._set_current_state(None, notify=False)

        self._started = False

    def process_event(self, event: Event) -> bool:
        """
        Process an event through the state machine.

        The machine evaluates all possible transitions from the current state
        (and its parent states), checks their guards, and executes the
        highest-priority valid transition if one exists.

        Args:
            event: The event to process

        Returns:
            bool: True if a transition was executed, False otherwise
        """
        if not self._started or not self._current_state:
            return False

        try:
            # Get potential transitions from both current state and its parent states
            potential_transitions = []
            current = self._current_state
            while current:
                transitions = self._graph.get_valid_transitions(current, event)
                potential_transitions.extend(transitions)
                current = current.parent

            if not potential_transitions:
                return False

            potential_transitions.sort(key=lambda t: t.get_priority(), reverse=True)

            # Evaluate guards to find valid transitions
            valid_transitions = []
            for transition in potential_transitions:
                # Check if all guards pass
                all_guards_pass = True
                for guard in transition.guards:
                    if not guard(event):
                        all_guards_pass = False
                        break
                if all_guards_pass:
                    valid_transitions.append(transition)

            if not valid_transitions:
                return False

            # Pick the highest-priority transition
            transition = valid_transitions[0]
            self._execute_transition(transition, event)

            # If target is a composite state, enter its initial state
            if isinstance(transition.target, CompositeState):
                initial_state = transition.target._initial_state
                if initial_state:
                    # Create and execute a transition to the initial state
                    initial_transition = Transition(
                        source=transition.target, target=initial_state, guards=[lambda e: True]
                    )
                    self._execute_transition(initial_transition, event)

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
            # Handle exit from current state
            self._handle_state_exit(transition)

            # Execute transition actions
            self._execute_transition_actions(transition, event)

            # Update current state and notify hooks
            self._update_state_and_notify(transition)

            # Handle entry to new state
            self._handle_state_entry(transition)

        except Exception as e:
            self._notify_error(e)
            raise

    def _handle_state_exit(self, transition: Transition) -> None:
        """Handle exiting current state and its ancestors up to common ancestor."""
        if not self._current_state:
            return

        source_ancestors = self._graph.get_composite_ancestors(self._current_state)
        target_ancestors = self._graph.get_composite_ancestors(transition.target)
        common_ancestor = next((s for s in source_ancestors if s in target_ancestors), None)

        if common_ancestor and common_ancestor == source_ancestors[0]:
            # Exit only current state if transitioning within same composite
            self._notify_exit(self._current_state)
            self._record_history_if_composite(self._current_state)
        else:
            # Exit up to but not including the common ancestor
            current = self._current_state
            while current and current != common_ancestor:
                self._notify_exit(current)
                self._record_history_if_composite(current)
                current = current.parent

    def _record_history_if_composite(self, state: State) -> None:
        """Record history state if parent is composite."""
        if isinstance(state.parent, CompositeState):
            self._graph.record_history(state.parent, state)

    def _execute_transition_actions(self, transition: Transition, event: Event) -> None:
        """Execute all actions associated with the transition."""
        for action in transition.actions:
            action(event)

    def _update_state_and_notify(self, transition: Transition) -> None:
        """Update current state and notify transition hooks."""
        self._set_current_state(transition.target, notify=False)
        for hook in self._hooks:
            if hasattr(hook, "on_transition"):
                hook.on_transition(transition.source, transition.target)

    def _handle_state_entry(self, transition: Transition) -> None:
        """Handle entering new state and its ancestors from common ancestor down."""
        source_ancestors = self._graph.get_composite_ancestors(self._current_state)
        target_ancestors = self._graph.get_composite_ancestors(transition.target)
        common_ancestor = next((s for s in source_ancestors if s in target_ancestors), None)

        # Enter only ancestors below the common ancestor
        for ancestor in target_ancestors:
            if ancestor == common_ancestor:
                break
            if ancestor not in source_ancestors:  # Only enter if not already active
                self._notify_enter(ancestor)
        self._notify_enter(transition.target)

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
        """
        Reset the state machine to its initial state.

        This clears all history states and restarts the machine if it was running.
        """
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

    def _validate_composite_states(self):
        """Ensure all composite states have initial states set."""
        def validate_composite(state):
            if isinstance(state, CompositeState):
                if not state.initial_state:
                    raise ValidationError(
                        f"CompositeState '{state.name}' has no initial state set"
                    )
                # Recursively validate children
                for child in state._children:
                    validate_composite(child)

        # Start validation from root
        validate_composite(self._initial_state)


class CompositeStateMachine(StateMachine):
    """
    An extended state machine that supports nested submachines.

    This allows complex state machines to be built by composing smaller,
    reusable state machines. Submachines can be integrated into the parent
    machine's state graph or kept separate.
    """

    def __init__(
        self,
        initial_state: State,
        validator: Optional[Validator] = None,
        hooks: Optional[List] = None,
        error_recovery: Optional[_ErrorRecoveryStrategy] = None,
    ):
        self._submachine_lock = Lock()
        self._state_lock = Lock()
        super().__init__(initial_state, validator, hooks, error_recovery)
        self._submachines = {}

    def add_submachine(self, state: CompositeState, submachine: "StateMachine") -> None:
        """
        Add a submachine under a composite state.

        This integrates all states and transitions from the submachine into
        the parent machine's graph, with the given composite state as their
        parent.

        Args:
            state: The composite state that will contain the submachine
            submachine: The state machine to integrate

        Raises:
            ValueError: If the provided state is not a CompositeState
        """
        with self._submachine_lock:
            if not isinstance(state, CompositeState):
                raise ValueError(f"State {state.name} must be a composite state")

            # First add the composite state if it's not already in the graph
            if state not in self._graph._nodes:
                self._graph.add_state(state)

            # Integrate submachine states into the same graph:
            for sub_state in submachine.get_states():
                # Add each state with the composite state as parent
                self._graph.add_state(sub_state, parent=state)

            # Integrate transitions
            for t in submachine.get_transitions():
                self._graph.add_transition(t)

            # Set the composite state's initial state to the submachine's initial state
            if submachine._initial_state:
                state._initial_state = submachine._initial_state
                # Add a transition from the composite state to its initial state
                self._graph.add_transition(
                    Transition(source=state, target=submachine._initial_state, guards=[lambda e: True])
                )

            self._submachines[state] = submachine

    def process_event(self, event: Event) -> bool:
        """
        Try to process event in the current submachine if the current state is composite.
        Otherwise, process normally in this machine.
        """
        if not self._started or not self._current_state:
            return False

        try:
            # Get potential transitions from both current state and its parent states
            potential_transitions = []
            current = self._current_state
            while current:
                # First check transitions from the current state
                transitions = self._graph.get_valid_transitions(current, event)
                potential_transitions.extend(transitions)

                # If we're in a submachine state, also check transitions from its parent composite state
                if current.parent in self._submachines:
                    composite_transitions = self._graph.get_valid_transitions(current.parent, event)
                    potential_transitions.extend(composite_transitions)

                current = current.parent

            if not potential_transitions:
                return False

            # Sort transitions by priority
            potential_transitions.sort(key=lambda t: t.get_priority(), reverse=True)

            # Evaluate guards to find valid transitions
            valid_transitions = []
            for transition in potential_transitions:
                # Check if all guards pass
                all_guards_pass = True
                for guard in transition.guards:
                    if not guard(event):
                        all_guards_pass = False
                        break
                if all_guards_pass:
                    valid_transitions.append(transition)

            if not valid_transitions:
                return False

            # Pick the highest-priority transition
            transition = valid_transitions[0]

            # Execute the transition
            self._execute_transition(transition, event)

            # If target is a composite state, enter its initial state or submachine's initial state
            if isinstance(transition.target, CompositeState):
                submachine = self._submachines.get(transition.target)
                if submachine:
                    initial_state = submachine._initial_state
                    if initial_state:
                        # Create and execute a transition to the initial state
                        initial_transition = Transition(
                            source=transition.target, target=initial_state, guards=[lambda e: True]
                        )
                        self._execute_transition(initial_transition, event)
                else:
                    initial_state = transition.target._initial_state
                    if initial_state:
                        # Create and execute a transition to the initial state
                        initial_transition = Transition(
                            source=transition.target, target=initial_state, guards=[lambda e: True]
                        )
                        self._execute_transition(initial_transition, event)

            return True

        except Exception as error:
            if self._error_recovery:
                self._error_recovery.recover(error, self)
            else:
                raise
            return False

    def start(self):
        with self._state_lock:
            with self._submachine_lock:
                super().start()
                # Initialize submachines in proper order
