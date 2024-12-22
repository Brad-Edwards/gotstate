# hsm/core/guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import Callable

from hsm.core.events import Event

class BasicGuards:
    """
    Provides simple guard checks as static methods. More complex conditions can be
    implemented as custom guards in plugins.
    """

    @staticmethod
    def check_condition(condition_fn: Callable[..., bool], **kwargs) -> bool:
        """
        Execute a condition function with given keyword arguments, returning True
        if the condition passes, False otherwise.

        :param condition_fn: A callable returning bool.
        :param kwargs: Additional parameters for the condition.
        """
        return condition_fn(**kwargs)


class _GuardAdapter:
    """
    Internal class adapting a simple callable to a GuardProtocol-like interface,
    ensuring consistent guard evaluation.
    """

    def __init__(self, guard_fn: Callable[["Event"], bool]) -> None:
        """
        Wrap a guard function which takes an Event and returns bool.

        :param guard_fn: A callable that takes an event and returns True/False.
        """
        self._guard_fn = guard_fn

    def check(self, event: "Event") -> bool:
        """
        Evaluate the wrapped guard function with the given event.

        :param event: The triggering event.
        :return: True if the guard condition passes, False otherwise.
        """
        return self._guard_fn(event)
