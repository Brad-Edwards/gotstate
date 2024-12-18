# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from hsm.core.actions import ActionExecutionError
from hsm.core.data_management import DataManager
from hsm.core.errors import (
    ConfigurationError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    InvalidTransitionError,
)
from hsm.core.hooks import HookManager
from hsm.core.states import CompositeState
from hsm.core.validation import ValidationSeverity, Validator
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractStateMachine, AbstractTransition
from hsm.interfaces.types import StateID

logger = logging.getLogger(__name__)


class StateMachine(AbstractStateMachine):
    """
    Core state machine implementation.

    Runtime Invariants:
    - Only one state is active at a time
    - State transitions are atomic
    - Event processing is sequential
    - State data is isolated between states and thread-safe
    - Hooks do not affect core behavior
    - Validation occurs before start

    Attributes:
        _states: Dictionary mapping state IDs to state objects
        _transitions: List of possible transitions
        _current_state: Currently active state
        _initial_state: Starting state
        _hook_manager: Manages state/transition hooks
        _data_manager: Manages state data
        _running: Whether machine is processing events
        _logger: Logger instance
    """

    def __init__(
        self,
        states: List[AbstractState],
        transitions: List[AbstractTransition],
        initial_state: AbstractState,
    ) -> None:
        """
        Initialize the state machine.

        Args:
            states: List of all possible states
            transitions: List of allowed transitions
            initial_state: Starting state

        Raises:
            ValueError: If states or transitions are empty or if initial state is not in the states list.
            TypeError: If arguments have wrong types.
        """
        if not states:
            raise ValueError("States list cannot be empty")
        # Transitions are allowed to be empty
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")

        # Initialize core components
        self._states: Dict[StateID, AbstractState] = {state.get_id(): state for state in states}
        self._transitions: List[AbstractTransition] = transitions
        self._initial_state: AbstractState = initial_state
        self._current_state: Optional[AbstractState] = None

        # Initialize managers
        self._hook_manager: HookManager = HookManager()
        self._data_manager: DataManager = DataManager()  # Now using DataManager

        # Initialize state
        self._running: bool = False
        self._logger: logging.Logger = logging.getLogger("hsm.core.state_machine")

        self._initialize_state_data()

    def _initialize_state_data(self) -> None:
        """Initialize the state data using the DataManager."""
        with self._data_manager.access_data() as data:
            for state_id in self._states:
                data[state_id] = {}

    def _validate_configuration(self) -> None:
        """
        Validate the state machine configuration.

        Raises:
            ConfigurationError: If validation fails
        """
        validator = Validator(list(self._states.values()), self._transitions, self._initial_state)

        # Check structure
        results = validator.validate_structure()
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR.name]
        if errors:
            raise ConfigurationError(
                "Invalid state machine structure", "structure", {r.message: r.context for r in errors}
            )

        # Check behavior
        results = validator.validate_behavior()
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR.name]
        if errors:
            raise ConfigurationError(
                "Invalid state machine behavior", "behavior", {r.message: r.context for r in errors}
            )

    def _execute_transition(self, transition: AbstractTransition, event: AbstractEvent) -> None:
        """Execute a transition."""

        source_state = transition.get_source_state()
        target_state = transition.get_target_state()

        with self._data_manager.access_data() as state_data:  # Use DataManager for thread-safety
            source_data = state_data[source_state.get_id()]
            target_data = state_data[target_state.get_id()]
            try:
                # Exit states
                self._exit_states(source_state, target_state, event, source_data)

                # Execute transition actions
                for action in transition.get_actions():
                    action.execute(event, source_data)

                # Enter states
                self._enter_states(source_state, target_state, event, target_data)

            except Exception as e:
                self._handle_operation_error(
                    "Transition",
                    e,
                    {
                        "source_state": source_state.get_id(),
                        "target_state": target_state.get_id(),
                        "event": event.get_id(),
                    },
                )

    def _exit_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """
        Helper function for exiting states during a transition.
        """
        current_state = self._current_state
        while current_state and current_state != source_state:
            self._hook_manager.call_on_exit(current_state.get_id())
            current_state.on_exit(event, data)
            if isinstance(current_state, CompositeState):
                current_state = current_state._parent_state if current_state._parent_state else None
            else:
                current_state = None

        if isinstance(source_state, CompositeState):
            is_target_substate = False
            if source_state.get_substates():
                for substate in source_state.get_substates():
                    if substate == target_state:
                        is_target_substate = True
                        break
            if source_state._current_substate:
                self._hook_manager.call_on_exit(source_state._current_substate.get_id())
                source_state._current_substate.on_exit(event, data)
                if not is_target_substate:
                    source_state._current_substate = None
            self._hook_manager.call_on_exit(source_state.get_id())
            source_state.on_exit(event, data)
            if not is_target_substate:
                self._current_state = None

    def _enter_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """
        Helper function for entering states during a transition.
        """

        # Build a path of states to enter from the source state to the target state
        path_to_target = []
        current_state = target_state
        while current_state != source_state:
            path_to_target.insert(0, current_state)
            if isinstance(current_state, CompositeState) and current_state._parent_state:
                current_state = current_state._parent_state
            else:
                current_state = None

        # Enter all states in the path
        for state in path_to_target:
            self._hook_manager.call_on_entry(state.get_id())
            state.on_entry(event, data)
            if isinstance(state, CompositeState):
                state._enter_substate(event, data)
                if state.has_history():
                    state.set_history_state(state._current_substate)

        self._current_state = target_state

    def _find_valid_transition(self, event: AbstractEvent) -> Optional[AbstractTransition]:
        """Find the highest priority valid transition for the current event."""
        if not self._current_state:
            return None

        valid_transitions = []

        for transition in self._transitions:
            if transition.get_source_state() == self._current_state or (
                isinstance(self._current_state, CompositeState)
                and transition.get_source_state() == self._current_state._current_substate
            ):
                guard = transition.get_guard()
                with self._data_manager.access_data() as state_data:  # Use DataManager for thread-safety
                    if guard:
                        try:
                            source_data = state_data[transition.get_source_state().get_id()]
                            if guard.check(event, source_data):
                                valid_transitions.append(transition)
                        except Exception as e:
                            logger.error("Guard check failed: %s", str(e))
                            continue
                    else:
                        valid_transitions.append(transition)

        if not valid_transitions:
            return None

        return max(valid_transitions, key=lambda t: t.get_priority())

    def get_state(self) -> Optional[AbstractState]:
        """
        Get the current state.

        Returns:
            Current state object or None if no current state

        """
        return self._current_state

    def start(self) -> None:
        """
        Start the state machine.

        Enters the initial state and begins processing events.

        Raises:
            InvalidStateError: If machine is already running
        """
        if self._running:
            raise InvalidStateError(
                "State machine already running", self.get_state().get_id() if self.get_state() else None, "start"
            )

        self._running = True
        self._current_state = self._initial_state
        # Initialize state data for all states
        self._initialize_state_data()
        with self._data_manager.access_data() as state_data:
            try:
                self._hook_manager.call_on_entry(self._current_state.get_id())
                self._current_state.on_entry(None, state_data[self._current_state.get_id()])
                if isinstance(self._current_state, CompositeState):
                    self._current_state._enter_substate(None, state_data[self._current_state.get_id()])
            except Exception as e:
                self._running = False
                raise InvalidStateError(
                    f"Failed to enter initial state: {str(e)}", self._initial_state.get_id(), "enter", {"error": str(e)}
                )

    def stop(self) -> None:
        """
        Stop the state machine.

        Exits current state and stops processing events.

        Raises:
            InvalidStateError: If machine is not running
        """
        if not self._running:
            raise InvalidStateError("State machine not running", None, "stop")

        if self._current_state:
            with self._data_manager.access_data() as state_data:
                try:
                    self._exit_states(
                        self._current_state, self._current_state, None, state_data[self._current_state.get_id()]
                    )
                except Exception as e:
                    self._logger.exception("Error during stop: %s", str(e))

        self._running = False
        self._current_state = None

    def process_event(self, event: AbstractEvent) -> None:
        """Process an event, potentially triggering a transition."""
        if not self._running:
            raise InvalidStateError("Cannot process events while stopped", None, "process_event")

        if not self._current_state:
            raise InvalidStateError("No current state", None, "process_event")

        # Find eligible transitions
        transition = self._find_valid_transition(event)

        if transition:
            # Execute transition
            self._execute_transition(transition, event)

    def get_current_state_id(self) -> StateID:
        """
        Get the ID of the current state.

        Returns:
            Current state ID or 'None' if no current state

        Raises:
            InvalidStateError: If no current state
        """
        if not self._current_state:
            raise InvalidStateError("No current state", "None", "get_current_state_id")
        return self._current_state.get_id()

    def _handle_operation_error(
        self, operation: str, error: Exception, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle errors that occur during state machine operations.

        Currently logs the error and raises it.

        Args:
            operation: The operation during which the error occurred.
            error: The exception object.
            details: Optional dictionary containing additional error details.
        """
        logger.error(f"Error during operation '{operation}': {error}", exc_info=True)
        if details:
            logger.error("Error details: %s", details)
        raise error  # Re-raise the original exception
