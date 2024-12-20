# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import Dict, List, Optional

from hsm.core.events import Event
from hsm.core.hooks import HookManager, HookProtocol
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition, _TransitionPrioritySorter
from hsm.core.validations import Validator


class _StateMachineContext:
    """
    Internal context management for StateMachine, tracking current state and
    available transitions. Not for direct use by library clients.
    """

    def __init__(self, initial_state: State) -> None:
        self._current_state = initial_state
        self._transitions: List[Transition] = []

    def get_current_state(self) -> State:
        return self._current_state

    def set_current_state(self, state: State) -> None:
        self._current_state = state

    def get_transitions(self) -> List[Transition]:
        return self._transitions

    def add_transition(self, transition: Transition) -> None:
        self._transitions.append(transition)


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
    A synchronous hierarchical state machine that manages state transitions
    based on incoming events. It starts in an initial state and processes events
    using defined transitions, optionally validated and hooked into by plugins.
    """

    def __init__(
        self,
        initial_state: State,
        validator: Optional[Validator] = None,
        hooks: Optional[List["HookProtocol"]] = None,
        error_recovery: Optional[_ErrorRecoveryStrategy] = None,
    ) -> None:
        """
        Initialize the state machine with an initial state, optional validator,
        hooks, and an error recovery strategy.

        :param initial_state: The starting state of the machine.
        :param validator: Optional Validator instance for checks.
        :param hooks: Optional list of hooks to execute on lifecycle events.
        :param error_recovery: Optional error recovery strategy.
        """
        self._context = _StateMachineContext(initial_state)
        self.validator = validator
        self._hooks = HookManager(hooks=hooks if hooks else [])
        self._error_recovery = error_recovery if error_recovery else _ErrorRecoveryStrategy()
        self._started = False
        self._stopped = False

    @property
    def current_state(self) -> State:
        """
        Return the current active state of the state machine.
        """
        return self._context.get_current_state()

    def start(self) -> None:
        """
        Start the state machine, executing initial entry actions and performing
        validation before beginning to process events.
        """
        if self._started:
            return
        if self.validator:
            self.validator.validate_state_machine(self)
        self.current_state.on_enter()
        self._hooks.execute_on_enter(self.current_state)
        self._started = True

    def process_event(self, event: Event) -> None:
        """
        Process an incoming event, performing state transitions if applicable.

        :param event: An Event instance triggering potential transitions.
        """
        if not self._started or self._stopped:
            # If not started or already stopped, ignore events.
            return

        transitions = self._context.get_transitions()
        valid_transitions = [t for t in transitions if t.source == self.current_state and t.evaluate_guards(event)]

        if not valid_transitions:
            # No transitions triggered by this event
            return

        # Pick the highest priority transition
        sorted_transitions = _TransitionPrioritySorter().sort(valid_transitions)
        chosen_transition = sorted_transitions[0]

        # Execute the transition: exit current, run actions, enter target
        try:
            self._hooks.execute_on_exit(self.current_state)
            self.current_state.on_exit()
            chosen_transition.execute_actions(event)
            self._context.set_current_state(chosen_transition.target)
            self.current_state.on_enter()
            self._hooks.execute_on_enter(self.current_state)
        except Exception as e:
            # If actions fail, attempt error recovery
            self._hooks.execute_on_error(e)
            self._error_recovery.recover(e, self)

    def stop(self) -> None:
        """
        Gracefully stop the state machine, executing exit actions and hooks.
        """
        if self._stopped:
            return
        self._hooks.execute_on_exit(self.current_state)
        self.current_state.on_exit()
        self._stopped = True

    # Public method to add transitions if needed
    def add_transition(self, transition: Transition) -> None:
        """
        Add a transition to the state machine. Useful if transitions are not all known at construction time.
        :param transition: The transition to add.
        """
        if self.validator:
            self.validator.validate_transition(transition)
        self._context.add_transition(transition)


class CompositeStateMachine(StateMachine):
    """
    A hierarchical extension of StateMachine that can contain nested submachines
    under composite states. This allows complex modeling of state hierarchies.
    """

    def __init__(
        self,
        initial_state: CompositeState,
        validator: Optional[Validator] = None,
        hooks: Optional[List["HookProtocol"]] = None,
        error_recovery: Optional[_ErrorRecoveryStrategy] = None,
    ) -> None:
        super().__init__(initial_state=initial_state, validator=validator, hooks=hooks, error_recovery=error_recovery)
        self._submachines: Dict[CompositeState, StateMachine] = {}

    def add_submachine(self, state: CompositeState, submachine: StateMachine) -> None:
        """
        Associate a subordinate StateMachine with a CompositeState, allowing
        nested behaviors within that state.

        :param state: The composite state serving as a container.
        :param submachine: The sub-state machine controlling nested states.
        """
        self._submachines[state] = submachine

    @property
    def submachines(self) -> Dict[CompositeState, StateMachine]:
        """
        Retrieve a mapping of CompositeStates to their associated submachines.
        """
        return self._submachines

    def start(self) -> None:
        super().start()
        # Optionally start submachines if desired. Implementation is minimal.
        for sm in self._submachines.values():
            if not sm.current_state:
                continue
            sm.start()

    def process_event(self, event: Event) -> None:
        super().process_event(event)
        # If the current state is a composite state with a submachine, we might
        # delegate events to the submachine as well.
        # This behavior can be customized as needed.
        if self.current_state in self._submachines:
            self._submachines[self.current_state].process_event(event)

    def stop(self) -> None:
        # Stop submachines first if needed
        for sm in self._submachines.values():
            sm.stop()
        super().stop()
