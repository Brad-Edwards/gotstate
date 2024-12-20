# hsm/core/actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import Callable

from hsm.core.events import Event


class BasicActions:
    """
    Provides simple, built-in action handlers. Users can extend or create custom
    actions in plugins.
    """

    @staticmethod
    def execute(action_fn: Callable[..., None], **kwargs) -> None:
        """
        Execute an action function with optional keyword arguments.

        :param action_fn: A callable representing the action's logic.
        :param kwargs: Additional parameters for the action.
        """
        action_fn(**kwargs)


class _ActionAdapter:
    """
    Internal adapter that wraps a user-defined callable into an ActionProtocol,
    ensuring consistent action invocation.
    """

    def __init__(self, action_fn: Callable[["Event"], None]) -> None:
        """
        Wrap an action function for consistent execution.
        :param action_fn: Function that takes an Event as a parameter.
        """
        self._action_fn = action_fn

    def run(self, event: "Event") -> None:
        """
        Execute the action for the given event.

        :param event: The triggering event.
        """
        self._action_fn(event)
