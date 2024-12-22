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
            # Get potential transitions from both current state and its parent states
            potential_transitions = []
            current = self._current_state
            while current:
                transitions = self._graph.get_valid_transitions(current, event)
                potential_transitions.extend(transitions)
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
            # First notify exit of current state and its ancestors up to source state
            if self._current_state:
                # Get the common ancestor between source and target states
                source_ancestors = self._graph.get_composite_ancestors(self._current_state)
                target_ancestors = self._graph.get_composite_ancestors(transition.target)
                common_ancestor = None
                for s in source_ancestors:
                    if s in target_ancestors:
                        common_ancestor = s
                        break

                # Exit up to but not including the common ancestor
                current = self._current_state
                while current and current != common_ancestor:
                    self._notify_exit(current)
                    if isinstance(current, CompositeState):
                        # Record history when exiting composite states
                        self._graph.record_history(current, self._current_state)
                    current = current.parent

            # Execute transition actions
            for action in transition.actions:
                action(event)

            # Update current state without notifications (they're handled here)
            self._set_current_state(transition.target, notify=False)

            # Notify enter of new state and its ancestors from common ancestor down
            target_ancestors = self._graph.get_composite_ancestors(transition.target)
            # Enter from common ancestor down to target state
            entered = set()
            for ancestor in reversed(target_ancestors):
                if ancestor not in entered and ancestor != common_ancestor:
                    self._notify_enter(ancestor)
                    entered.add(ancestor)
            self._notify_enter(transition.target)

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
            # First try to process in the current submachine if we're in a composite state
            if isinstance(self.current_state, CompositeState):
                submachine = self._submachines.get(self.current_state)
                if submachine and submachine.process_event(event):
                    return True

            # Get potential transitions from the graph
            potential_transitions = self._graph.get_valid_transitions(self._current_state, event)
            if not potential_transitions:
                return False

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

            # Record history for composite states before leaving them
            if isinstance(self._current_state.parent, CompositeState):
                self._graph.record_history(self._current_state.parent, self._current_state)

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
