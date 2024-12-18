# hsm/core/actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator

from hsm.core.errors import ActionExecutionError
from hsm.interfaces.abc import AbstractAction
from hsm.interfaces.protocols import Event


class BasicAction(AbstractAction):
    """
    A base action class that raises NotImplementedError.

    Runtime Invariants:
    - Calling execute() will always raise NotImplementedError until overridden.

    Example:
        action = BasicAction()
        action.execute(event, state_data)  # Raises NotImplementedError
    """

    def _handle_execution_error(self, message: str, state_data: Any, event: Event) -> None:
        """Common method for handling execution errors."""
        raise ActionExecutionError(message, action_name=self.__class__.__name__, state_data=state_data, event=event)

    def execute(self, event: Event, state_data: Any) -> None:
        raise NotImplementedError("BasicAction must be subclassed and execute() overridden.")


class NoOpAction(BasicAction):
    """
    An action that does nothing.

    This is useful as a placeholder action or for testing transitions that require
    an action but do not need to modify state.

    Runtime Invariants:
    - execute() completes instantly without side-effects.

    Example:
        action = NoOpAction()
        action.execute(event, state_data)  # Does nothing
    """

    def execute(self, event: Event, state_data: Any) -> None:
        # No operation performed
        pass


class LoggingAction(BasicAction):
    """
    An action that logs the event and state_data for debugging.

    Runtime Invariants:
    - Logging is atomic and does not modify state_data or event.
    - No exceptions are raised unless logging fails at a system level.

    Attributes:
        logger_name: Name of the logger to use.

    Example:
        action = LoggingAction(logger_name="hsm.action")
        action.execute(event, state_data)
        # Logs event ID and state_data content.
    """

    def __init__(self, logger_name: str = "hsm.actions"):
        self.logger = logging.getLogger(logger_name)

    def execute(self, event: Event, state_data: Any) -> None:
        if event is None:
            raise AttributeError("Event cannot be None")
        event_id = event.get_id()
        self.logger.info(f"Executing action for event: {event_id}, state_data: {state_data}")


class SetDataAction(BasicAction):
    """
    An action that sets a key-value pair in the state_data atomically.

    Uses a context manager to revert changes if any error occurs.

    Runtime Invariants:
    - Changes to state_data are atomic.
    - If an error occurs during execute(), changes are reverted.

    Attributes:
        key: The key to set in state_data.
        value: The value to set.

    Example:
        action = SetDataAction("status", "updated")
        action.execute(event, state_data)
        # state_data["status"] is set to "updated"
    """

    def __init__(self, key: str, value: Any):
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        self.key = key
        self.value = value

    @contextmanager
    def _temporary_change(self, state_data: Dict[str, Any]) -> Generator[None, None, None]:
        original = state_data.get(self.key, None)
        state_data[self.key] = self.value
        try:
            yield
        except Exception:
            # Revert changes if an error occurs
            if original is None:
                del state_data[self.key]
            else:
                state_data[self.key] = original
            raise

    def execute(self, event: Event, state_data: Any) -> None:
        if not isinstance(state_data, dict):
            self._handle_execution_error("State data must be a dictionary", state_data, event)
        with self._temporary_change(state_data):
            # Potentially do more work here. If errors occur, changes revert.
            # No additional errors here, so changes persist.
            pass


class ValidateDataAction(BasicAction):
    """
    An action that validates the state_data against certain criteria.
    Raises ActionExecutionError if validation fails.

    Runtime Invariants:
    - Validation is deterministic.
    - If validation fails, no changes are made to state_data.

    Attributes:
        required_keys: A list of keys that must be present in state_data.
        condition: A callable that returns True if validation passes, False otherwise.

    Example:
        def check_condition(data):
            return data.get("counter", 0) > 0

        action = ValidateDataAction(["counter"], check_condition)
        action.execute(event, state_data)
        # Raises ActionExecutionError if "counter" not in state_data or condition fails.
    """

    def __init__(self, required_keys: list[str], condition: Any):
        if required_keys is None:
            raise ValueError("required_keys cannot be None")
        if not isinstance(required_keys, list):
            raise TypeError("required_keys must be a list")
        if condition is None:
            raise ValueError("condition cannot be None")
        if not callable(condition):
            raise TypeError("condition must be callable")
        self.required_keys = required_keys
        self.condition = condition

    def execute(self, event: Event, state_data: Any) -> None:
        if not isinstance(state_data, dict):
            self._handle_execution_error("Invalid state_data type", state_data, event)

        for key in self.required_keys:
            if key not in state_data:
                self._handle_execution_error(f"Missing required key: {key}", state_data, event)

        try:
            state_data_copy = state_data.copy()
            if not self.condition(state_data_copy):
                self._handle_execution_error("Validation condition failed", state_data, event)
        except Exception as e:
            self._handle_execution_error(f"Error during condition evaluation: {str(e)}", state_data, event)
