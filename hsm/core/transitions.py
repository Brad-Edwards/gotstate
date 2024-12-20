# hsm/core/transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hsm.core.actions import ActionProtocol
    from hsm.core.events import Event
    from hsm.core.guards import GuardProtocol
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
        guards: list["GuardProtocol"] = None,
        actions: list["ActionProtocol"] = None,
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
        raise NotImplementedError()

    def evaluate_guards(self, event: "Event") -> bool:
        """
        Evaluate the attached guards to determine if the transition can occur.

        :param event: The triggering event.
        :return: True if all guards pass, otherwise False.
        """
        raise NotImplementedError()

    def execute_actions(self, event: "Event") -> None:
        """
        Execute the transition's actions, if any, when moving to the target state.

        :param event: The triggering event.
        """
        raise NotImplementedError()

    def get_priority(self) -> int:
        """
        Return the priority level assigned to this transition.
        """
        raise NotImplementedError()

    @property
    def source(self) -> "State":
        """
        The source state of the transition.
        """
        raise NotImplementedError()

    @property
    def target(self) -> "State":
        """
        The target state of the transition.
        """
        raise NotImplementedError()


class _TransitionPrioritySorter:
    """
    Internal utility to sort a list of transitions by their priority, ensuring
    that the highest priority valid transition is selected first.
    """

    def sort(self, transitions: list[Transition]) -> list[Transition]:
        """
        Sort and return transitions ordered by priority (descending or ascending as required).
        """
        raise NotImplementedError()


class _GuardEvaluator:
    """
    Internal helper to evaluate a list of guard conditions against an event.
    """

    def evaluate(self, guards: list["GuardProtocol"], event: "Event") -> bool:
        """
        Check all guards. Return True if all pass, False if any fail.
        """
        raise NotImplementedError()


class _ActionExecutor:
    """
    Internal helper to execute a list of actions when a transition fires, handling
    errors and ensuring consistent execution order.
    """

    def execute(self, actions: list["ActionProtocol"], event: "Event") -> None:
        """
        Run the given actions for the event.
        """
        raise NotImplementedError()
