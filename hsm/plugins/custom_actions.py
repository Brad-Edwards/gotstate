# hsm/plugins/custom_actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.events import Event


class MyCustomAction:
    """
    A user-defined action that executes a custom function when triggered.
    """

    def __init__(self, action_fn: callable) -> None:
        """
        Initialize with an action function.
        """
        self.action_fn = action_fn

    def execute(self, event: Event) -> None:
        """
        Execute the custom action with the given event.
        """
        self.action_fn(event)

    # Add alias for compatibility
    run = execute
