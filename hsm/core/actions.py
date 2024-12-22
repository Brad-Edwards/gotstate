# hsm/core/actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import Callable

from hsm.core.events import Event


class BasicActions:
    """
    Provides simple, built-in action handlers for state machine transitions and events.

    This class offers basic action handling functionality that can be used directly
    or extended through plugins. Actions are functions that are executed when states
    are entered, exited, or when specific events occur.

    Example:
        ```python
        def my_action(value: str):
            print(f"Executing action with {value}")

        # Execute a simple action
        BasicActions.execute(my_action, value="hello")
        ```
    """

    @staticmethod
    def execute(action_fn: Callable[..., None], **kwargs) -> None:
        """
        Execute an action function with optional keyword arguments.

        This method provides a standardized way to execute action functions within
        the state machine. It supports any callable that accepts keyword arguments.

        Args:
            action_fn: A callable function that implements the action's logic.
                      The function can accept any number of keyword arguments.
            **kwargs: Variable keyword arguments that will be passed to the action function.
                     These can include state data, event data, or any other contextual information.

        Example:
            ```python
            def log_state_change(old_state: str, new_state: str):
                print(f"Transitioning from {old_state} to {new_state}")

            BasicActions.execute(log_state_change,
                               old_state="idle",
                               new_state="running")
            ```
        """
        action_fn(**kwargs)


class _ActionAdapter:
    """
    Internal adapter that wraps a user-defined callable into an ActionProtocol.

    This adapter ensures consistent action invocation within the state machine by
    standardizing how actions handle events. It's primarily used internally by the
    state machine implementation.

    Note:
        This is an internal class and should not be used directly by library users.
        Instead, use the BasicActions class or create custom action handlers.
    """

    def __init__(self, action_fn: Callable[["Event"], None]) -> None:
        """
        Initialize the action adapter with a function that processes events.

        Args:
            action_fn: A function that takes an Event object as its parameter.
                      This function defines the action's behavior when triggered
                      by an event in the state machine.
        """
        self._action_fn = action_fn

    def run(self, event: "Event") -> None:
        """
        Execute the action for the given event.

        Args:
            event: The Event object that triggered this action. Contains
                  event-specific data and context for the action to process.
        """
        self._action_fn(event)
