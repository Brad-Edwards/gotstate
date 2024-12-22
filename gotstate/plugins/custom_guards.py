# hsm/plugins/custom_guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from gotstate.core.events import Event


class MyCustomGuard:
    """
    A custom guard that takes a condition function and evaluates it against events.
    """

    def __init__(self, condition_fn: callable) -> None:
        """
        Initialize with a condition function.
        """
        self.condition_fn = condition_fn

    def check(self, event: Event) -> bool:
        """
        Check if the guard condition is satisfied.

        Args:
            event: The event to check against

        Returns:
            bool: True if condition is satisfied, False otherwise
        """
        return self.condition_fn(event)
