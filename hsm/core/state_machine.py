# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
from threading import Lock
from typing import List, Optional, Set

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
    Base state machine implementation that manages states and transitions.
    Supports hierarchical state nesting and composite states.
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
        self._graph.set_initial_state(None, initial_state)  # Set as root initial state
        self._graph.set_current_state(initial_state)  # Set initial state as current state

    @property
    def current_state(self) -> Optional[State]:
        """Get the current state of the state machine."""
        return self._graph.get_current_state()

    def start(self) -> None:
        """Start the state machine."""
        with self._transition_lock:
            if self._started:
                return

            # Validate before entering state
            errors = self._graph.validate()
            if errors:
                raise ValidationError("\n".join(errors))

            self._validator.validate_state_machine(self)

            # Resolve initial or historical active state
            initial_state = self._graph.get_initial_state(None)  # Get root initial state
            if initial_state:
                resolved_state = self._graph.resolve_active_state(initial_state)
                self._set_current_state(resolved_state)

                # Build path from root to resolved state
                state_path = []
                current = resolved_state
                while current:
                    state_path.append(current)
                    current = current.parent

                # Enter states from outermost to innermost
                for state in reversed(state_path):
                    self._notify_enter(state)

            self._started = True

    def stop(self) -> None:
        """Stop the state machine."""
        with self._transition_lock:
            if not self._started:
                return

            if self._graph.get_current_state():
                # Record history for all composite ancestors before stopping
                for composite in self._graph.get_composite_ancestors(self._graph.get_current_state()):
                    self._graph.record_history(composite, self._graph.get_current_state())
                self._notify_exit(self._graph.get_current_state())
                self._set_current_state(None)

            self._started = False

    def process_event(self, event: Event) -> bool:
        """
        Process an event by finding and executing a valid transition.

        Args:
            event: The event to process

        Returns:
            True if event was handled, False otherwise
        """
        if not self._started:
            return False

        with self._transition_lock:
            valid_transitions = []
            if self.current_state:  # Only process transitions if we have a current state
                # First try transitions from current state
                for transition in self._graph.get_valid_transitions(self.current_state, event):
                    # If there are no guards, the transition is always valid
                    if not transition.guards:
                        valid_transitions.append(transition)
                        continue

                    # Check if any guard explicitly matches the event
                    has_matching_guard = False
                    for guard in transition.guards:
                        try:
                            if guard(event):
                                has_matching_guard = True
                                break
                        except:
                            # If guard evaluation fails, skip this guard
                            continue

                    if has_matching_guard:
                        valid_transitions.append(transition)

                # If no transitions found, try parent states
                if not valid_transitions:
                    current = self.current_state.parent
                    while current and not valid_transitions:
                        for transition in self._graph.get_valid_transitions(current, event):
                            # If there are no guards, the transition is always valid
                            if not transition.guards:
                                valid_transitions.append(transition)
                                continue

                            # Check guards for parent state transitions too
                            has_matching_guard = False
                            for guard in transition.guards:
                                try:
                                    if guard(event):
                                        has_matching_guard = True
                                        break
                                except:
                                    # If guard evaluation fails, skip this guard
                                    continue

                            if has_matching_guard:
                                valid_transitions.append(transition)
                        current = current.parent

            if not valid_transitions:
                return False

            # Pick highest-priority transition
            transition = max(valid_transitions, key=lambda t: t.get_priority())
            result = self._execute_transition(transition, event)

            # If result is None or False, the transition was not handled or failed
            if result is None or result is False:
                return False

            # If result is True, the transition was successful
            return True

    def _execute_transition(self, transition: Transition, event: Event) -> Optional[bool]:
        """
        Execute a transition between states.

        Args:
            transition: The transition to execute
            event: The triggering event

        Returns:
            True if successful, False if failed but handled, None if not handled

        Raises:
            Exception: If an error occurs during transition and no error recovery is provided
        """
        previous_state = self._graph.get_current_state()
        try:
            # Find the common ancestor between source and target states
            source_ancestors = []
            current = previous_state
            while current:
                source_ancestors.append(current)
                current = current.parent

            target_ancestors = []
            current = transition.target
            while current:
                target_ancestors.append(current)
                current = current.parent

            # Find common ancestor
            common_ancestor = None
            for state in source_ancestors:
                if state in target_ancestors:
                    common_ancestor = state
                    break

            # Exit up to common ancestor
            current = previous_state
            while current and current != common_ancestor:
                self._notify_exit(current)
                current = current.parent

            # Execute transition actions
            for action in transition.actions:
                action(event)

            # Update current state without notifications (we'll handle them)
            self._graph.set_current_state(transition.target)

            # Record history for all composite ancestors
            for composite in self._graph.get_composite_ancestors(transition.target):
                self._graph.record_history(composite, transition.target)

            # Notify transition
            for hook in self._hooks:
                if hasattr(hook, "on_transition"):
                    hook.on_transition(transition.source, transition.target)

            # Enter from common ancestor to target, including parent states
            target_path = []
            current = transition.target
            while current and current != common_ancestor:
                target_path.append(current)
                current = current.parent

            # Enter states from outermost to innermost
            for state in reversed(target_path):
                self._notify_enter(state)

            # Handle composite state initial transitions after entering all states
            for state in reversed(target_path):
                if isinstance(state, CompositeState) and state not in source_ancestors:
                    initial_state = self._graph.get_initial_state(state)
                    if initial_state:
                        # Create and execute a transition to the initial state
                        initial_transition = Transition(source=state, target=initial_state, guards=[lambda e: True])
                        self._execute_transition(initial_transition, event)

            return True

        except Exception as e:
            # Restore previous state if we failed during transition
            self._graph.set_current_state(previous_state)
            self._notify_error(e)
            if self._error_recovery:
                self._error_recovery.recover(e, self)
                return False
            # Re-raise the exception if no error recovery is provided
            raise

    def _set_current_state(self, state: Optional[State]) -> None:
        """Set the current state."""
        self._graph.set_current_state(state)

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
        if isinstance(parent, CompositeState) and not self._graph.get_initial_state(parent):
            self._graph.set_initial_state(parent, state)

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
        self._set_current_state(self._graph.get_initial_state(None))
        if was_started:
            self.start()

    def validate(self) -> List[str]:
        """Expose the graph's validation results."""
        return self._graph.validate()


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
        self._machine_lock = Lock()  # Single lock for all machine operations
        self._submachines = {}
        super().__init__(initial_state, validator, hooks, error_recovery)

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
            ValueError: If the provided state is not in the machine or not a composite state
        """
        with self._machine_lock:
            # First add the composite state if it's not already in the graph
            if state not in self._graph._nodes:
                self._graph.add_state(state)

            if not isinstance(state, CompositeState):
                raise ValueError(f"State {state.name} must be a composite state")

            # Merge the submachine's graph into our graph
            self._graph.merge_submachine(state, submachine._graph)

            # Store reference to submachine
            self._submachines[state] = submachine

            # Set initial state if not already set
            submachine_initial = submachine._graph.get_initial_state(None)  # Get root initial state
            if submachine_initial and not self._graph.get_initial_state(state):
                mapped_initial = self._graph._nodes[submachine_initial].state
                self._graph.set_initial_state(state, mapped_initial)

    def start(self):
        """Start the state machine and all submachines."""
        # Try to acquire the lock with a timeout to prevent deadlock
        if not self._machine_lock.acquire(timeout=2.0):
            raise RuntimeError("Failed to acquire lock for machine start - possible deadlock")

        try:
            # If already started, just return
            if self._started:
                return

            # First start the main machine
            super().start()

            # Then initialize all submachines in the proper order
            for composite_state, submachine in self._submachines.items():
                if not submachine._started:
                    # Copy the initial state from submachine if needed
                    submachine_initial = submachine._graph.get_initial_state(None)  # Get root initial state
                    if submachine_initial and not self._graph.get_initial_state(composite_state):
                        mapped_initial = self._graph._nodes[submachine_initial].state
                        self._graph.set_initial_state(composite_state, mapped_initial)
                    submachine._started = True
        finally:
            self._machine_lock.release()
