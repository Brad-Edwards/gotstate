# hsm/core/guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.events import Event


class BasicGuards:
    """
    Provides simple guard checks as static methods. More complex conditions can be
    implemented as custom guards in plugins.
    """

    @staticmethod
    def check_condition(condition_fn: callable, **kwargs) -> bool:
        """
        Execute a condition function with given keyword arguments, returning True
        if the condition passes, False otherwise.

        :param condition_fn: A callable returning bool.
        :param kwargs: Additional parameters for the condition.
        """
        raise NotImplementedError()


class _GuardAdapter:
    """
    Internal class adapting a simple callable to the GuardProtocol interface,
    ensuring consistent guard evaluation.
    """

    def __init__(self, guard_fn: callable) -> None:
        """
        Wrap a guard function.
        """
        raise NotImplementedError()

    def check(self, event: "Event") -> bool:
        """
        Evaluate the wrapped guard function with the given event.
        """
        raise NotImplementedError()
