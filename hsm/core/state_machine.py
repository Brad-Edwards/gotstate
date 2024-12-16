# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set

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
from hsm.core.validation import ValidationSeverity, Validator
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractStateMachine, AbstractTransition
from hsm.interfaces.types import StateID


class StateMachine(AbstractStateMachine):
    """
    Core state machine implementation.

    Runtime Invariants:
    - Only one state is active at a time
    - State transitions are atomic
    - Event processing is sequential
    - State data is isolated between states
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
            ConfigurationError: If configuration is invalid
            ValueError: If states or transitions are empty
            TypeError: If arguments have wrong types
        """
        if not states:
            raise ValueError("States list cannot be empty")
        if not transitions:
            raise ValueError("Transitions list cannot be empty")
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")

        # Initialize core components
        self._states: Dict[StateID, AbstractState] = {state.get_id(): state for state in states}
        self._transitions = transitions
        self._initial_state = initial_state
        self._current_state: Optional[AbstractState] = None

        # Initialize managers
        self._hook_manager = HookManager()
        self._data_manager = DataManager()

        # Initialize state
        self._running = False
        self._logger = logging.getLogger("hsm.core.state_machine")

        # Validate configuration
        self._validate_configuration()

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

    @contextmanager
    def _transition_context(self, transition: AbstractTransition) -> Any:
        """
        Context manager for atomic transitions.

        Handles:
        - Hook notifications
        - State entry/exit
        - Error recovery
        - Data cleanup

        Args:
            transition: The transition being executed

        Yields:
            None

        Raises:
            InvalidTransitionError: If transition fails
        """
        source_id = transition.get_source_state_id()
        target_id = transition.get_target_state_id()
        old_state = self._current_state

        try:
            # Pre-transition hooks
            self._hook_manager.call_pre_transition(transition)

            # Exit current state
            if old_state:
                old_state.on_exit()

            yield

            # Enter new state
            target_state = self._states[target_id]
            self._current_state = target_state  # Set new state before enter
            target_state.on_enter()

            # Post-transition hooks
            self._hook_manager.call_post_transition(transition)

        except Exception as e:
            # Attempt recovery
            self._logger.exception("Transition failed from %s to %s: %s", source_id, target_id, str(e))
            # Restore previous state
            self._current_state = old_state
            if old_state:
                try:
                    old_state.on_enter()  # Re-enter previous state
                except Exception:
                    self._logger.exception("Recovery failed")
            raise InvalidTransitionError(f"Transition failed: {str(e)}", source_id, target_id, None, {"error": str(e)})

    def start(self) -> None:
        """
        Start the state machine.

        Enters the initial state and begins processing events.

        Raises:
            InvalidStateError: If machine is already running
        """
        if self._running:
            raise InvalidStateError("State machine already running", self.get_current_state_id(), "start")

        self._running = True
        self._current_state = self._initial_state
        try:
            self._current_state.on_enter()
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
            raise InvalidStateError(
                "State machine not running", None, "stop"  # or "None" if StateID is str  # This is the operation name
            )

        if self._current_state:
            try:
                self._current_state.on_exit()
            except Exception as e:
                self._logger.exception("Error during stop: %s", str(e))

        self._running = False
        self._current_state = None

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        was_running = self._running

        if self._current_state:
            try:
                self._current_state.on_exit()
            except Exception as e:
                self._logger.exception("Error during state exit in reset: %s", str(e))

        # Clear all state data
        for state in self._states.values():
            state.data.clear()

        self._current_state = None
        self._running = False

        if was_running:
            self.start()

    def process_event(self, event: AbstractEvent) -> None:
        """Process an event, potentially triggering a transition."""
        if not self._running:
            raise InvalidStateError("Cannot process events while stopped", None, "process_event")

        if not self._current_state:
            raise InvalidStateError("No current state", None, "process_event")

        # Find eligible transitions
        current_id = self._current_state.get_id()
        eligible = [t for t in self._transitions if t.get_source_state_id() == current_id]

        # Check guards
        valid = []
        for transition in eligible:
            guard = transition.get_guard()
            try:
                if guard is None or guard.check(event, self._current_state.data):
                    valid.append(transition)
            except Exception as e:
                raise GuardEvaluationError(
                    f"Guard check failed: {str(e)}", str(guard), self._current_state.data, event, {"error": str(e)}
                )

        if not valid:
            return  # No valid transitions

        # Select highest priority transition
        transition = max(valid, key=lambda t: t.get_priority())

        # Execute transition
        try:
            with self._transition_context(transition):
                # Execute actions
                for action in transition.get_actions():
                    try:
                        action.execute(event, self._current_state.data)
                    except Exception as e:
                        raise ActionExecutionError(
                            f"Action execution failed: {str(e)}",
                            str(action),
                            self._current_state.data,
                            event,
                            {"error": str(e)},
                        )
        except (ActionExecutionError, GuardEvaluationError) as e:
            raise InvalidTransitionError(
                str(e), transition.get_source_state_id(), transition.get_target_state_id(), event, e.details
            )

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
