# hsm/plugins/custom_guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.events import Event


class MyCustomGuard:
    """
    A user-defined guard that evaluates a custom condition when checked.
    """

    def __init__(self, condition_fn: callable) -> None:
        """
        Initialize with a condition function.
        """
        self.condition_fn = condition_fn

    def check(self, event: Event) -> bool:
        """
        Check the guard condition with the given event.
        Returns True if the condition is met, False otherwise.
        """
        return self.condition_fn(event)
