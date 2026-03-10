"""
State machine orchestration and lifecycle management.

Orchestrates state machine components, manages machine lifecycle,
coordinates core component interactions, and handles dynamic modifications.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import icontract

from gotstate.core.event import Event, EventQueue
from gotstate.core.region import Region, RegionManager
from gotstate.core.state import CompositeState, State, StateType
from gotstate.core.transition import Transition, TransitionKind
from gotstate.exceptions import (
    MachineAlreadyRunningError,
    MachineError,
    MachineNotInitializedError,
)

logger = logging.getLogger(__name__)


class MachineStatus(Enum):
    """Defines the possible states of a state machine."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    MODIFYING = auto()
    TERMINATING = auto()
    TERMINATED = auto()


@icontract.invariant(lambda self: isinstance(self._status, MachineStatus), "Machine status must be valid")
class StateMachine:
    """Represents a hierarchical state machine.

    Coordinates all components and manages the machine lifecycle.
    Provides run-to-completion event processing semantics.

    Class Invariants:
    1. Machine status must be a valid MachineStatus
    """

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    def __init__(self, name: str) -> None:
        self._name = name
        self._status = MachineStatus.UNINITIALIZED
        self._states: Dict[str, State] = {}
        self._transitions: List[Transition] = []
        self._event_queue = EventQueue()
        self._region_manager = RegionManager()
        self._current_state: Optional[State] = None
        self._initial_state: Optional[State] = None
        self._lock = threading.RLock()
        self._on_transition_callbacks: List[Callable[[Transition], None]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> MachineStatus:
        return self._status

    @property
    def current_state(self) -> Optional[State]:
        return self._current_state

    @property
    def states(self) -> Dict[str, State]:
        return dict(self._states)

    @property
    def transitions(self) -> List[Transition]:
        return list(self._transitions)

    @property
    def event_queue(self) -> EventQueue:
        return self._event_queue

    @icontract.require(lambda state: state is not None, "State must not be None")
    def add_state(self, state: State) -> None:
        """Register a state with the machine."""
        with self._lock:
            if state.name in self._states:
                raise MachineError(f"State '{state.name}' already exists in machine '{self._name}'")
            self._states[state.name] = state

    @icontract.require(lambda transition: transition is not None, "Transition must not be None")
    def add_transition(self, transition: Transition) -> None:
        """Register a transition with the machine."""
        with self._lock:
            self._transitions.append(transition)

    @icontract.require(lambda state: state is not None, "Initial state must not be None")
    def set_initial_state(self, state: State) -> None:
        """Designate the initial state of the machine."""
        if state.name not in self._states:
            raise MachineError(f"State '{state.name}' is not registered with machine '{self._name}'")
        self._initial_state = state

    def start(self) -> None:
        """Initialize and start the state machine.

        Enters the initial state and begins event processing.
        """
        with self._lock:
            if self._status == MachineStatus.ACTIVE:
                raise MachineAlreadyRunningError(f"Machine '{self._name}' is already running")
            if self._initial_state is None:
                raise MachineNotInitializedError(f"Machine '{self._name}' has no initial state")
            self._status = MachineStatus.INITIALIZING
            self._current_state = self._initial_state
            self._initial_state.enter()
            self._status = MachineStatus.ACTIVE

    def stop(self) -> None:
        """Stop the state machine and exit all active states."""
        with self._lock:
            if self._status != MachineStatus.ACTIVE:
                return
            self._status = MachineStatus.TERMINATING
            if self._current_state is not None:
                self._current_state.exit()
                self._current_state = None
            self._region_manager.deactivate_all()
            self._event_queue.clear()
            self._status = MachineStatus.TERMINATED

    def process_event(self, event: Event) -> bool:
        """Process an event through the state machine.

        Implements run-to-completion: finds an enabled transition,
        executes it, and updates the current state.

        Returns True if a transition was fired, False otherwise.
        """
        with self._lock:
            if self._status != MachineStatus.ACTIVE:
                raise MachineNotInitializedError(f"Machine '{self._name}' is not active")

            enabled = self._find_enabled_transitions(event)
            if not enabled:
                self._event_queue.enqueue(event)
                return False

            transition = enabled[0]
            transition.execute(event)
            event.consume()

            if transition.target is not None and transition.kind != TransitionKind.INTERNAL:
                self._current_state = transition.target

            for callback in self._on_transition_callbacks:
                try:
                    callback(transition)
                except Exception:
                    logger.exception("Transition callback failed")

            return True

    def _find_enabled_transitions(self, event: Event) -> List[Transition]:
        """Find all transitions enabled by the given event, sorted by priority."""
        enabled: List[Transition] = []
        for transition in self._transitions:
            if transition.source is self._current_state and transition.is_enabled(event):
                enabled.append(transition)
        enabled.sort()
        return enabled

    def on_transition(self, callback: Callable[[Transition], None]) -> None:
        """Register a callback invoked after each transition."""
        self._on_transition_callbacks.append(callback)

    def add_region(self, region: Region) -> None:
        """Add a parallel region to the machine."""
        self._region_manager.add_region(region)

    def get_state(self, name: str) -> State:
        """Get a registered state by name."""
        if name not in self._states:
            raise MachineError(f"State '{name}' not found in machine '{self._name}'")
        return self._states[name]

    def __repr__(self) -> str:
        current = self._current_state.name if self._current_state else "None"
        return f"StateMachine(name={self._name!r}, status={self._status.name}, current={current})"


class ProtocolMachine(StateMachine):
    """A state machine that enforces protocol constraints on operation sequences."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._allowed_operations: Dict[str, List[str]] = {}

    def allow_operation(self, state_name: str, operation: str) -> None:
        """Register that an operation is allowed in a given state."""
        if state_name not in self._allowed_operations:
            self._allowed_operations[state_name] = []
        self._allowed_operations[state_name].append(operation)

    def is_operation_allowed(self, operation: str) -> bool:
        """Check if an operation is allowed in the current state."""
        if self._current_state is None:
            return False
        allowed = self._allowed_operations.get(self._current_state.name, [])
        return operation in allowed


class SubmachineMachine(StateMachine):
    """A reusable state machine component that can be referenced by other machines."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._entry_points: Dict[str, State] = {}
        self._exit_points: Dict[str, State] = {}

    def add_entry_point(self, name: str, state: State) -> None:
        self._entry_points[name] = state

    def add_exit_point(self, name: str, state: State) -> None:
        self._exit_points[name] = state

    @property
    def entry_points(self) -> Dict[str, State]:
        return dict(self._entry_points)

    @property
    def exit_points(self) -> Dict[str, State]:
        return dict(self._exit_points)


class MachineBuilder:
    """Builds state machine configurations using the Builder pattern."""

    def __init__(self, name: str) -> None:
        self._machine = StateMachine(name)
        self._built = False

    def add_state(self, name: str, state_type: StateType = StateType.SIMPLE) -> State:
        state = State(name, state_type)
        self._machine.add_state(state)
        return state

    def set_initial(self, state: State) -> "MachineBuilder":
        self._machine.set_initial_state(state)
        return self

    def add_transition(
        self,
        source: State,
        target: State,
        trigger: Optional[str] = None,
        guard: Optional[Callable[..., bool]] = None,
        action: Optional[Callable[..., None]] = None,
    ) -> Transition:
        transition = Transition(source, target, TransitionKind.EXTERNAL, guard, action, trigger)
        self._machine.add_transition(transition)
        return transition

    @icontract.ensure(lambda result: result is not None, "Build must return a machine")
    def build(self) -> StateMachine:
        """Finalize and return the configured state machine."""
        if self._built:
            raise MachineError("Machine has already been built")
        self._built = True
        return self._machine


class MachineModifier:
    """Manages dynamic state machine modifications with atomicity guarantees."""

    def __init__(self, machine: StateMachine) -> None:
        self._machine = machine
        self._pending_states: List[State] = []
        self._pending_transitions: List[Transition] = []

    def stage_add_state(self, state: State) -> None:
        self._pending_states.append(state)

    def stage_add_transition(self, transition: Transition) -> None:
        self._pending_transitions.append(transition)

    def apply(self) -> None:
        """Apply all staged modifications atomically."""
        with self._machine._lock:
            old_status = self._machine._status
            self._machine._status = MachineStatus.MODIFYING
            try:
                for state in self._pending_states:
                    self._machine.add_state(state)
                for transition in self._pending_transitions:
                    self._machine.add_transition(transition)
                self._pending_states.clear()
                self._pending_transitions.clear()
            except Exception:
                self._machine._status = old_status
                raise
            self._machine._status = old_status

    def rollback(self) -> None:
        """Discard all staged modifications."""
        self._pending_states.clear()
        self._pending_transitions.clear()


class MachineMonitor:
    """Monitors state machine execution and collects metrics."""

    def __init__(self, machine: StateMachine) -> None:
        self._machine = machine
        self._transition_count = 0
        self._event_count = 0
        self._state_history: List[str] = []
        machine.on_transition(self._on_transition)

    def _on_transition(self, transition: Transition) -> None:
        self._transition_count += 1
        if transition.target is not None:
            self._state_history.append(transition.target.name)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def state_history(self) -> List[str]:
        return list(self._state_history)
