# hsm/plugins/custom_guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.events import Event


class MyCustomGuard:
    """
    A user-defined guard example that checks a custom condition before allowing a transition.
    """

    def __init__(self, condition_fn: callable) -> None:
        """
        Initialize with a condition function returning bool.
        """
        raise NotImplementedError()

    def check(self, event: "Event") -> bool:
        """
        Evaluate the custom condition with the given event.

        :param event: The triggering event.
        :return: True if allowed, False if not.
        """
        raise NotImplementedError()
