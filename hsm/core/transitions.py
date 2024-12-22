# hsm/core/transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from hsm.core.errors import TransitionError
from hsm.core.events import Event

if TYPE_CHECKING:
    from hsm.core.states import State

# Type aliases for clarity
GuardFunction = Union[Callable[[Event], bool], Callable[[Event], asyncio.Future[bool]]]
ActionFunction = Union[Callable[[Event], None], Callable[[Event], asyncio.Future[None]]]


class Transition:
    """
    Defines a possible path from one state to another, guarded by conditions and
    potentially performing actions. Supports both synchronous and asynchronous guards/actions.
    """

    def __init__(
        self,
        source: "State",
        target: "State",
        guards: Optional[List[GuardFunction]] = None,
        actions: Optional[List[ActionFunction]] = None,
        priority: int = 0,
    ) -> None:
        self._source = source
        self._target = target
        self._guards = guards if guards else []
        self._actions = actions if actions else []
        self._priority = priority

    async def evaluate_guards(self, event: Event) -> bool:
        """
        Evaluate the attached guards to determine if the transition can occur.
        Supports both sync and async guard functions.

        :param event: The triggering event.
        :return: True if all guards pass, otherwise False.
        """
        return await _GuardEvaluator().evaluate(self._guards, event)

    async def execute_actions(self, event: Event) -> None:
        """
        Execute the transition's actions, if any, when moving to the target state.
        Supports both sync and async action functions.

        :param event: The triggering event.
        :raises TransitionError: If any action fails.
        """
        try:
            await _ActionExecutor().execute(self._actions, event)
        except Exception as e:
            raise TransitionError(f"Action execution failed: {e}")

    def get_priority(self) -> int:
        """Return the priority level assigned to this transition."""
        return self._priority

    @property
    def source(self) -> "State":
        """The source state of the transition."""
        return self._source

    @property
    def target(self) -> "State":
        """The target state of the transition."""
        return self._target

    @property
    def guards(self) -> List[GuardFunction]:
        """The guard conditions for this transition."""
        return self._guards

    @property
    def actions(self) -> List[ActionFunction]:
        """The actions to execute when this transition occurs."""
        return self._actions


class _TransitionPrioritySorter:
    """Internal utility to sort transitions by priority."""

    def sort(self, transitions: List[Transition]) -> List[Transition]:
        return sorted(transitions, key=lambda t: t.get_priority(), reverse=True)


class _GuardEvaluator:
    """Internal helper to evaluate guard conditions against an event."""

    async def evaluate(self, guards: List[GuardFunction], event: Event) -> bool:
        """
        Check all guards. Handles both sync and async guard functions.

        :param guards: List of guard callables.
        :param event: The event to evaluate against.
        :return: True if all guards return True, otherwise False.
        """
        for guard in guards:
            if inspect.iscoroutinefunction(guard):
                if not await guard(event):
                    return False
            else:
                if not guard(event):
                    return False
        return True


class _ActionExecutor:
    """Internal helper to execute transition actions."""

    async def execute(self, actions: List[ActionFunction], event: Event) -> None:
        """
        Run the given actions for the event. Handles both sync and async actions.

        :param actions: List of action callables.
        :param event: The triggering event.
        :raises Exception: If any action fails.
        """
        for action in actions:
            if inspect.iscoroutinefunction(action):
                await action(event)
            else:
                action(event)
