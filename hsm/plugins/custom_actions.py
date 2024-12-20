# hsm/plugins/custom_actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.events import Event


class MyCustomAction:
    """
    A user-defined action example that executes a custom piece of code during a transition.
    """

    def __init__(self, action_fn: callable) -> None:
        """
        Initialize with a custom action function.
        """
        raise NotImplementedError()

    def run(self, event: "Event") -> None:
        """
        Perform the action in response to the given event.

        :param event: The triggering event.
        """
        raise NotImplementedError()
