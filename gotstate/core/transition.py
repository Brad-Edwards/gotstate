"""
Transition types and behavior management.

Implements the transition type hierarchy, guard conditions, actions,
and execution semantics for state changes.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Callable, List, Optional

import icontract

from gotstate.core.event import Event
from gotstate.core.state import State
from gotstate.exceptions import GuardError, InvalidTransitionError

logger = logging.getLogger(__name__)


class TransitionKind(Enum):
    """Defines the different types of transitions."""

    EXTERNAL = auto()
    INTERNAL = auto()
    LOCAL = auto()
    COMPOUND = auto()


class TransitionPriority(Enum):
    """Defines priority levels for transition conflict resolution."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@icontract.invariant(lambda self: isinstance(self._kind, TransitionKind), "Transition kind must be valid")
@icontract.invariant(lambda self: self._source is not None, "Source state must not be None")
class Transition:
    """Represents a transition between states in a hierarchical state machine.

    Encapsulates guard conditions, actions, source/target states, and
    trigger events. Guards must be side-effect free.

    Class Invariants:
    1. Transition kind must be a valid TransitionKind
    2. Source state must not be None
    """

    @icontract.require(lambda source: source is not None, "Source state is required")
    @icontract.require(lambda kind: isinstance(kind, TransitionKind), "kind must be a valid TransitionKind")
    def __init__(
        self,
        source: State,
        target: Optional[State],
        kind: TransitionKind = TransitionKind.EXTERNAL,
        guard: Optional[Callable[..., bool]] = None,
        action: Optional[Callable[..., None]] = None,
        trigger: Optional[str] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL,
    ) -> None:
        self._source = source
        self._target = target
        self._kind = kind
        self._guard = guard
        self._action = action
        self._trigger = trigger
        self._priority = priority

    @property
    def source(self) -> State:
        return self._source

    @property
    def target(self) -> Optional[State]:
        return self._target

    @property
    def kind(self) -> TransitionKind:
        return self._kind

    @property
    def guard(self) -> Optional[Callable[..., bool]]:
        return self._guard

    @property
    def action(self) -> Optional[Callable[..., None]]:
        return self._action

    @property
    def trigger(self) -> Optional[str]:
        return self._trigger

    @property
    def priority(self) -> TransitionPriority:
        return self._priority

    def is_enabled(self, event: Optional[Event] = None) -> bool:
        """Check if this transition is enabled.

        A transition is enabled if:
        1. The trigger matches the event name (or trigger is None for completion)
        2. The guard condition evaluates to True (or no guard is set)
        """
        if self._trigger is not None:
            if event is None or event.name != self._trigger:
                return False
        elif event is not None:
            return False

        if self._guard is not None:
            try:
                return bool(self._guard())
            except Exception as e:
                raise GuardError(f"Guard evaluation failed on transition from '{self._source.name}': {e}") from e
        return True

    def execute(self, event: Optional[Event] = None) -> None:
        """Execute the transition: exit source, run action, enter target.

        For EXTERNAL transitions: exits source, runs action, enters target.
        For INTERNAL transitions: runs action only (no exit/entry).
        For LOCAL transitions: minimizes exit/entry scope.
        """
        if self._kind == TransitionKind.INTERNAL:
            self._run_action(event)
            return

        if self._kind == TransitionKind.EXTERNAL:
            self._source.exit()
            self._run_action(event)
            if self._target is not None:
                self._target.enter()
            return

        if self._kind == TransitionKind.LOCAL:
            lca = State.find_lca(self._source, self._target) if self._target else None
            if lca is self._source or lca is self._target:
                self._run_action(event)
                if self._target is not None and self._target is not self._source:
                    if self._source.is_active:
                        self._source.exit()
                    self._target.enter()
            else:
                self._source.exit()
                self._run_action(event)
                if self._target is not None:
                    self._target.enter()

    def _run_action(self, event: Optional[Event] = None) -> None:
        if self._action is not None:
            try:
                self._action()
            except Exception:
                logger.exception("Transition action failed on %s -> %s", self._source.name, self._target)
                raise

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return NotImplemented
        return self._priority.value < other._priority.value

    def __repr__(self) -> str:
        target_name = self._target.name if self._target else "None"
        return (
            f"Transition({self._source.name} -> {target_name}, " f"kind={self._kind.name}, trigger={self._trigger!r})"
        )


class ExternalTransition(Transition):
    """External transition that fully exits source and enters target."""

    def __init__(
        self,
        source: State,
        target: State,
        guard: Optional[Callable[..., bool]] = None,
        action: Optional[Callable[..., None]] = None,
        trigger: Optional[str] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL,
    ) -> None:
        if target is None:
            raise InvalidTransitionError("External transition must have a target state")
        super().__init__(source, target, TransitionKind.EXTERNAL, guard, action, trigger, priority)


class InternalTransition(Transition):
    """Internal transition that executes without exiting or entering states.

    Source and target must be the same state.
    """

    def __init__(
        self,
        state: State,
        guard: Optional[Callable[..., bool]] = None,
        action: Optional[Callable[..., None]] = None,
        trigger: Optional[str] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL,
    ) -> None:
        super().__init__(state, state, TransitionKind.INTERNAL, guard, action, trigger, priority)


class LocalTransition(Transition):
    """Local transition that minimizes state exit/entry within a composite state."""

    def __init__(
        self,
        source: State,
        target: State,
        guard: Optional[Callable[..., bool]] = None,
        action: Optional[Callable[..., None]] = None,
        trigger: Optional[str] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL,
    ) -> None:
        super().__init__(source, target, TransitionKind.LOCAL, guard, action, trigger, priority)


class CompoundTransition(Transition):
    """Compound transition composed of multiple segments through pseudostates."""

    def __init__(
        self,
        source: State,
        target: Optional[State],
        segments: Optional[List[Transition]] = None,
        trigger: Optional[str] = None,
        priority: TransitionPriority = TransitionPriority.NORMAL,
    ) -> None:
        super().__init__(source, target, TransitionKind.COMPOUND, trigger=trigger, priority=priority)
        self._segments: List[Transition] = list(segments) if segments else []

    @property
    def segments(self) -> List[Transition]:
        return list(self._segments)

    def add_segment(self, segment: Transition) -> None:
        self._segments.append(segment)

    def execute(self, event: Optional[Event] = None) -> None:
        """Execute all segments in order."""
        for segment in self._segments:
            if not segment.is_enabled(event):
                return
            segment.execute(event)


class ProtocolTransition(Transition):
    """Transition that enforces protocol constraints on operation sequences."""

    def __init__(
        self,
        source: State,
        target: State,
        guard: Optional[Callable[..., bool]] = None,
        action: Optional[Callable[..., None]] = None,
        trigger: Optional[str] = None,
        pre_condition: Optional[Callable[[], bool]] = None,
        post_condition: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(source, target, TransitionKind.EXTERNAL, guard, action, trigger)
        self._pre_condition = pre_condition
        self._post_condition = post_condition

    def is_enabled(self, event: Optional[Event] = None) -> bool:
        if not super().is_enabled(event):
            return False
        if self._pre_condition is not None and not self._pre_condition():
            return False
        return True

    def execute(self, event: Optional[Event] = None) -> None:
        super().execute(event)
        if self._post_condition is not None and not self._post_condition():
            raise InvalidTransitionError("Protocol post-condition violated")


class TimeTransition(Transition):
    """Transition triggered by time events."""

    @icontract.require(lambda duration: duration > 0, "Duration must be positive")
    def __init__(
        self,
        source: State,
        target: State,
        duration: float,
        action: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__(source, target, TransitionKind.EXTERNAL, action=action)
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration


class ChangeTransition(Transition):
    """Transition triggered by a boolean condition becoming True."""

    @icontract.require(lambda condition: callable(condition), "Condition must be callable")
    def __init__(
        self,
        source: State,
        target: State,
        condition: Callable[[], bool],
        action: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__(source, target, TransitionKind.EXTERNAL, guard=condition, action=action)
        self._condition = condition

    @property
    def condition(self) -> Callable[[], bool]:
        return self._condition
