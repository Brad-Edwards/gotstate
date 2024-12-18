# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

from hsm.core.errors import HSMError, InvalidStateError
from hsm.core.state_machine import StateMachine
from hsm.interfaces.abc import (
    AbstractAction,
    AbstractCompositeState,
    AbstractEvent,
    AbstractGuard,
    AbstractState,
    AbstractStateMachine,
    AbstractTransition,
)
from hsm.interfaces.protocols import Event
from hsm.interfaces.types import EventID, StateID
from hsm.runtime.event_queue import AsyncEventQueue, EventQueueError
from hsm.runtime.timers import AsyncTimer

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncLockManager:
    """Manages async locks and operations."""

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}

    async def with_lock(self, name: str, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute operation with lock protection."""
        lock = self._locks.get(name) or self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            return await operation()


class AsyncHSMError(HSMError):
    """Base exception for async state machine errors."""

    # Meant to be subclassed by other errors
    pass


class AsyncStateError(AsyncHSMError):
    """Raised when async state operations fail."""

    def __init__(self, message: str, state_id: StateID, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.state_id = state_id
        self.details = details or {}


class AsyncTransitionError(AsyncHSMError):
    """Raised when async transitions fail."""

    def __init__(
        self,
        message: str,
        source_state: AbstractState,
        target_state: AbstractState,
        event: AbstractEvent,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.source_state = source_state
        self.target_state = target_state
        self.event = event
        self.details = details or {}


@runtime_checkable
class AsyncGuard(Protocol):
    """Protocol for async guard conditions."""

    async def check(self, event: Event, state_data: Any) -> bool: ...


@runtime_checkable
class AsyncAction(Protocol):
    """Protocol for async actions."""

    async def execute(self, event: Event, state_data: Any) -> None: ...


class AsyncState(AbstractState):
    """Base class for async states."""

    def __init__(self, state_id: StateID):
        if not state_id:
            raise ValueError("Invalid state ID in state initialization")
        self._state_id = state_id
        self._entry_lock = asyncio.Lock()
        self._exit_lock = asyncio.Lock()

    async def on_entry(self, event: AbstractEvent, data: Any) -> None:
        """Async state entry handler."""
        async with self._entry_lock:
            try:
                await self._do_enter(event, data)
            except Exception as e:
                raise AsyncStateError(f"Error during state entry: {str(e)}", self._state_id, {"error": str(e)}) from e

    async def _do_enter(self, event: AbstractEvent, data: Any) -> None:
        """Override this method to implement custom entry logic."""
        # Meant to be overridden by subclasses
        pass

    async def on_exit(self, event: AbstractEvent, data: Any) -> None:
        """Async state exit handler."""
        async with self._exit_lock:
            try:
                await self._do_exit(event, data)
            except Exception as e:
                raise AsyncStateError(f"Error during state exit: {str(e)}", self._state_id, {"error": str(e)}) from e

    async def _do_exit(self, event: AbstractEvent, data: Any) -> None:
        """Override this method to implement custom exit logic."""
        pass

    def get_id(self) -> StateID:
        """Get state identifier."""
        return self._state_id


class AsyncCompositeState(AsyncState, AbstractCompositeState):
    def __init__(
        self,
        state_id: StateID,
        substates: List[AbstractState],
        initial_state: Optional[AbstractState] = None,
        has_history: bool = False,
        parent_state: Optional[AbstractState] = None,
    ):
        super().__init__(state_id)

        if initial_state is None and substates:
            initial_state = substates[0]

        self._substates = substates
        self._initial_state = initial_state
        self._has_history = has_history
        self._history_state: Optional[AbstractState] = None
        self._current_substate: Optional[AbstractState] = None
        self._parent_state = parent_state

        # Validate initial state based on whether we have substates
        if substates:
            if initial_state is None:
                raise ValueError("initial_state cannot be None when substates exist")
            if initial_state not in substates:
                raise ValueError("initial_state must be one of the substates")
        else:
            if initial_state is not None:
                raise ValueError("initial_state must be None when substates is empty")

    @property
    def parent_state(self) -> Optional[AbstractState]:
        return self._parent_state

    @parent_state.setter
    def parent_state(self, value: AbstractState) -> None:
        self._parent_state = value

    async def on_entry(self, event: AbstractEvent, data: Any) -> None:
        await self._enter_substate(event, data)

    async def on_exit(self, event: AbstractEvent, data: Any) -> None:
        if self._current_substate:
            await self._current_substate.on_exit(event, data)

    async def _enter_substate(self, event: AbstractEvent, data: Any) -> None:
        """Enter the appropriate substate."""
        if self.has_history() and self._history_state:
            self._current_substate = self._history_state
        else:
            self._current_substate = self.get_initial_state()

        if self._current_substate:
            await self._current_substate.on_entry(event, data)

    def get_substates(self) -> List[AbstractState]:
        return self._substates

    def get_initial_state(self) -> AbstractState:
        if self._initial_state is None:
            raise ValueError("Initial state not set for composite state")
        return self._initial_state

    def has_history(self) -> bool:
        return self._has_history

    def set_history_state(self, state: AbstractState) -> None:
        if state not in self._substates:
            raise ValueError("State is not a substate of this composite state")
        self._history_state = state


class AsyncTransition(AbstractTransition):
    """Async transition implementation."""

    def __init__(
        self,
        source: AbstractState,
        target: AbstractState,
        guard: Optional[AsyncGuard] = None,
        actions: Optional[List[AsyncAction]] = None,
        priority: int = 0,
    ):
        if source is None or target is None:
            raise ValueError("Source and target states cannot be None")
        self._source = source
        self._target = target
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state(self) -> AbstractState:
        return self._source

    def get_target_state(self) -> AbstractState:
        return self._target

    def get_guard(self) -> Optional[AsyncGuard]:
        return self._guard

    def get_actions(self) -> List[AsyncAction]:
        return self._actions

    def get_priority(self) -> int:
        return self._priority


class AsyncStateMachine(AbstractStateMachine):
    """Async variant of the state machine implementation."""

    def __init__(
        self,
        states: List[AsyncState],
        transitions: List[AsyncTransition],
        initial_state: AsyncState,
        max_queue_size: Optional[int] = None,
    ):
        """Initialize async state machine."""
        if not states:
            raise ValueError("States list cannot be empty")
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")

        self._states: Dict[StateID, AsyncState] = {state.get_id(): state for state in states}
        self._transitions: List[AsyncTransition] = transitions
        self._initial_state: AsyncState = initial_state
        self._current_state: Optional[AsyncState] = None
        self._event_queue: AsyncEventQueue = AsyncEventQueue(max_size=max_queue_size)
        self._state_data: Dict[StateID, Any] = {}
        self._initialize_state_data()

        # Initialize timer with default callback
        def timer_callback(timer_id: str, event: AbstractEvent) -> None:
            if self._running and self._current_state:
                asyncio.create_task(self.process_event(event))

        self._timer: AsyncTimer = AsyncTimer("state_machine_timer", timer_callback)
        self._running: bool = False
        self._state_changes: Set[StateID] = set()
        self._state_change_callbacks: List[Callable[[StateID, StateID], None]] = []
        self._lock_manager: AsyncLockManager = AsyncLockManager()

    def _initialize_state_data(self) -> None:
        """Initialize the _state_data dictionary."""
        self._state_data = {}
        for state_id, state in self._states.items():
            self._state_data[state_id] = {}

    def _get_state_data(self, state_id: StateID) -> Any:
        """Get the data for the given state."""
        if state_id not in self._state_data:
            raise ValueError(f"No data found for state: {state_id}")
        return self._state_data[state_id]

    async def start(self) -> None:
        """Start the state machine."""
        if self._running:
            return

        self._running = True
        self._current_state = self._initial_state
        try:
            await self._current_state.on_entry(None, self._get_state_data(self._current_state.get_id()))
            if isinstance(self._current_state, AsyncCompositeState):
                await self._current_state._enter_substate(None, self._get_state_data(self._current_state.get_id()))
            asyncio.create_task(self._process_events())
        except Exception as e:
            self._running = False
            raise AsyncHSMError(f"Failed to start state machine: {str(e)}") from e

    async def stop(self) -> None:
        """Stop the state machine."""
        if not self._running:
            return

        self._running = False
        try:
            if self._current_state:
                await self._current_state.on_exit(None, self._get_state_data(self._current_state.get_id()))
                if isinstance(self._current_state, AsyncCompositeState):
                    current_substate = self._current_state._current_substate
                    if current_substate:
                        await current_substate.on_exit(None, self._get_state_data(current_substate.get_id()))
            await self._event_queue.shutdown()
            await self._timer.shutdown()
            # Wait for event processing loop to complete
            await asyncio.sleep(0)  # Give the event loop a chance to finish
        except Exception as e:
            logger.error("Error during state machine shutdown: %s", str(e))
            raise AsyncHSMError(f"Failed to stop state machine: {str(e)}") from e

    async def process_event(self, event: AbstractEvent) -> None:
        """Process an event asynchronously."""
        if not self._running:
            raise AsyncHSMError("State machine is not running")

        try:
            await self._event_queue.enqueue(event)
        except EventQueueError as e:
            raise AsyncHSMError(f"Failed to enqueue event: {str(e)}") from e

    async def _process_events(self) -> None:
        """Event processing loop."""
        while self._running:
            try:
                event = await self._event_queue.dequeue()
                if event:  # Check if we got a valid event
                    await self._lock_manager.with_lock("processing", lambda e=event: self._handle_event(e))
            except EventQueueError:
                # Queue being empty is a normal condition, just continue
                await asyncio.sleep(0)  # Yield control to other tasks
            except Exception as e:
                logger.error("Unexpected error in event processing loop: %s", str(e))
                logger.exception("Stack trace:")
                await asyncio.sleep(0)  # Yield control to other tasks

    async def _handle_event(self, event: AbstractEvent) -> None:
        """Handle a single event."""
        if not self._current_state:
            return

        # Get the transition first
        transition = await self._lock_manager.with_lock("transition", lambda: self._find_valid_transition(event))

        if transition:
            await self._execute_transition(transition, event)

    async def _find_valid_transition(self, event: AbstractEvent) -> Optional[AsyncTransition]:
        """Find the highest priority valid transition for the current event."""
        valid_transitions = []
        current_state_id = self._current_state.get_id()

        for transition in self._transitions:
            if transition.get_source_state().get_id() != current_state_id:
                continue

            guard = transition.get_guard()
            if guard:
                try:
                    if await guard.check(event, self._get_state_data(transition.get_source_state().get_id())):
                        valid_transitions.append(transition)
                except Exception as e:
                    logger.error("Guard check failed: %s", str(e))
                    continue
            else:
                valid_transitions.append(transition)

        if not valid_transitions:
            return None

        return max(valid_transitions, key=lambda t: t.get_priority())

    async def _execute_transition(self, transition: AsyncTransition, event: AbstractEvent) -> None:
        """Execute a transition."""
        if not self._current_state:
            return

        source_state = transition.get_source_state()
        target_state = transition.get_target_state()
        source_data = self._get_state_data(source_state.get_id())
        target_data = self._get_state_data(target_state.get_id())

        try:
            # Exit states
            await self._exit_states(source_state, target_state, event, source_data)

            # Execute transition actions
            for action in transition.get_actions():
                await action.execute(event, source_data)
                # Enter states
            await self._enter_states(source_state, target_state, event, target_data)

        except Exception as e:
            raise AsyncTransitionError(
                f"Error during transition execution: {str(e)}", source_state, target_state, event
            ) from e

    async def _exit_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """
        Helper function for exiting states during a transition.
        """
        current_state = self._current_state
        while current_state and current_state != source_state:
            await current_state.on_exit(event, self._get_state_data(current_state.get_id()))
            if isinstance(current_state, AsyncCompositeState):
                current_state = current_state._parent_state if current_state._parent_state else None
            else:
                current_state = None

        if isinstance(source_state, AsyncCompositeState):
            if source_state._current_substate:
                await source_state._current_substate.on_exit(event, data)
        await source_state.on_exit(event, data)
        self._current_state = None

    async def _enter_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """
        Helper function for entering states during a transition.
        """
        # Build a path of states to enter from the source state to the target state
        path_to_target = []
        current_state = target_state
        while current_state != source_state:
            path_to_target.insert(0, current_state)
            if isinstance(current_state, AsyncCompositeState) and current_state._parent_state:
                current_state = current_state._parent_state
            else:
                current_state = None

        # Enter all states in the path
        for state in path_to_target:
            await state.on_entry(event, self._get_state_data(state.get_id()))
            if isinstance(state, AsyncCompositeState):
                await state._enter_substate(event, data)
                if state.has_history():
                    state.set_history_state(state._current_substate)

        self._current_state = target_state

    def get_state(self) -> Optional[AsyncState]:
        """Get the current state."""
        return self._current_state

    def is_running(self) -> bool:
        """Check if the state machine is running."""
        return self._running

    def add_state_change_callback(self, callback: Callable[[StateID, StateID], None]) -> None:
        self._state_change_callbacks.append(callback)

    def _notify_state_change(self, source_id: StateID, target_id: StateID) -> None:
        """Notify all callbacks of a state change."""
        for callback in self._state_change_callbacks:
            callback(source_id, target_id)

    def get_current_state_id(self) -> StateID:
        """
        Get the ID of the current state.

        Returns:
            Current state ID or 'None' if no current state

        Raises:
            InvalidStateError: If no current state
        """
        if not self._current_state:
            raise InvalidStateError("No current state", "None", "get_current_state_id")
        return self._current_state.get_id()

    async def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the state machine."""
        return {
            "current_state": self._current_state.get_id() if self._current_state else None,
            "running": self._running,
            "state_changes": list(self._state_changes),
            "queue_size": await self._event_queue.size(),
            "states": list(self._states.keys()),
            "transitions": [
                {
                    "source": t.get_source_state().get_id(),
                    "target": t.get_target_state().get_id(),
                    "priority": t.get_priority(),
                }
                for t in self._transitions
            ],
        }
