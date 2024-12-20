# hsm/core/transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import Callable, List, Optional

from hsm.core.errors import TransitionError
from hsm.core.events import Event
from hsm.core.states import State


class Transition:
    """
    Defines a possible path from one state to another, guarded by conditions and
    potentially performing actions. Used by the state machine to change states
    when events are processed.
    """

    def __init__(
        self,
        source: "State",
        target: "State",
        guards: Optional[List[Callable[[Event], bool]]] = None,
        actions: Optional[List[Callable[[Event], None]]] = None,
        priority: int = 0,
    ) -> None:
        """
        Initialize a transition with a source and target state, optional guards,
        actions, and a priority used if multiple transitions are possible.

        :param source: The origin State of this transition.
        :param target: The destination State of this transition.
        :param guards: Guard conditions that must be true for the transition.
        :param actions: Actions to execute when the transition occurs.
        :param priority: Numeric priority; higher priority transitions are chosen first.
        """
        self._source = source
        self._target = target
        self._guards = guards if guards else []
        self._actions = actions if actions else []
        self._priority = priority

    def evaluate_guards(self, event: Event) -> bool:
        """
        Evaluate the attached guards to determine if the transition can occur.

        :param event: The triggering event.
        :return: True if all guards pass, otherwise False.
        """
        return _GuardEvaluator().evaluate(self._guards, event)

    def execute_actions(self, event: Event) -> None:
        """
        Execute the transition's actions, if any, when moving to the target state.

        :param event: The triggering event.
        :raises TransitionError: If any action fails.
        """
        try:
            _ActionExecutor().execute(self._actions, event)
        except Exception as e:
            raise TransitionError(f"Action execution failed: {e}")

    def get_priority(self) -> int:
        """
        Return the priority level assigned to this transition.
        """
        return self._priority

    @property
    def source(self) -> "State":
        """
        The source state of the transition.
        """
        return self._source

    @property
    def target(self) -> "State":
        """
        The target state of the transition.
        """
        return self._target

    @property
    def guards(self) -> List[Callable[[Event], bool]]:
        """The guard conditions for this transition."""
        return self._guards

    @property
    def actions(self) -> List[Callable[[Event], None]]:
        """The actions to execute when this transition occurs."""
        return self._actions


class _TransitionPrioritySorter:
    """
    Internal utility to sort a list of transitions by their priority, ensuring
    that the highest priority valid transition is selected first.
    """

    def sort(self, transitions: List[Transition]) -> List[Transition]:
        """
        Sort and return transitions ordered by priority, highest first.

        :param transitions: A list of Transition instances.
        :return: A sorted list of Transition instances by descending priority.
        """
        return sorted(transitions, key=lambda t: t.get_priority(), reverse=True)


class _GuardEvaluator:
    """
    Internal helper to evaluate a list of guard conditions against an event.
    """

    def evaluate(self, guards: List[Callable[[Event], bool]], event: Event) -> bool:
        """
        Check all guards. Return True if all pass, False if any fail.

        :param guards: List of guard callables.
        :param event: The event to evaluate against.
        :return: True if all guards return True, otherwise False.
        """
        for g in guards:
            if not g(event):
                return False
        return True


class _ActionExecutor:
    """
    Internal helper to execute a list of actions when a transition fires, handling
    errors and ensuring consistent execution order.
    """

    def execute(self, actions: List[Callable[[Event], None]], event: Event) -> None:
        """
        Run the given actions for the event.

        :param actions: List of action callables.
        :param event: The triggering event.
        :raises Exception: If any action fails.
        """
        for a in actions:
            a(event)
