# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
from contextlib import contextmanager
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
    TypeVar,
    runtime_checkable,
)

from hsm.core.errors import HSMError
from hsm.core.state_machine import StateMachine
from hsm.interfaces.abc import AbstractAction, AbstractEvent, AbstractGuard, AbstractState, AbstractTransition
from hsm.interfaces.protocols import Event
from hsm.interfaces.types import EventID, StateID
from hsm.runtime.event_queue import AsyncEventQueue, EventQueueError
from hsm.runtime.timers import AsyncTimer

logger = logging.getLogger(__name__)

# Type variable for generic async operations
T = TypeVar("T")

# Forward references
if TYPE_CHECKING:
    from typing import Type

    AsyncTransitionType = Type["AsyncTransition"]
    AsyncStateType = Type["AsyncState"]


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

    pass


class AsyncHSMBase:
    """Base class with common validation and error handling."""

    def _validate_state_id(self, state_id: StateID, context: str) -> None:
        """Validate state ID."""
        if not state_id:
            raise ValueError(f"Invalid state ID in {context}")

    def _validate_transition(self, transition: "AsyncTransition", states: Dict[str, "AsyncState"]) -> None:
        """Validate transition references valid states."""
        source_id = transition.get_source_state_id()
        target_id = transition.get_target_state_id()

        if source_id not in states:
            raise ValueError(f"Invalid source state: {source_id}")
        if target_id not in states:
            raise ValueError(f"Invalid target state: {target_id}")

    @staticmethod
    def wrap_error(e: Exception, context: str, **kwargs: Any) -> AsyncHSMError:
        """Wrap exception with context."""
        return AsyncHSMError(f"Error in {context}: {str(e)}", **kwargs)


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
        source_id: StateID,
        target_id: StateID,
        event: AbstractEvent,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.source_id = source_id
        self.target_id = target_id
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


class AsyncState(AsyncHSMBase, AbstractState):
    """Base class for async states."""

    def __init__(self, state_id: StateID):
        self._validate_state_id(state_id, "state initialization")
        self._state_id = state_id
        self._data: Dict[str, Any] = {}
        self._entry_lock = asyncio.Lock()
        self._exit_lock = asyncio.Lock()

    async def on_enter(self) -> None:
        """Async state entry handler."""
        async with self._entry_lock:
            try:
                await self._do_enter()
            except Exception as e:
                raise AsyncStateError(f"Error during state entry: {str(e)}", self._state_id, {"error": str(e)}) from e

    async def _do_enter(self) -> None:
        """Override this method to implement custom entry logic."""
        pass

    async def on_exit(self) -> None:
        """Async state exit handler."""
        async with self._exit_lock:
            try:
                await self._do_exit()
            except Exception as e:
                raise AsyncStateError(f"Error during state exit: {str(e)}", self._state_id, {"error": str(e)}) from e

    async def _do_exit(self) -> None:
        """Override this method to implement custom exit logic."""
        pass

    @property
    def data(self) -> Dict[str, Any]:
        """Access state data dictionary."""
        return self._data.copy()

    def get_id(self) -> StateID:
        """Get state identifier."""
        return self._state_id


class AsyncTransition(AbstractTransition):
    """Async transition implementation."""

    def __init__(
        self,
        source_id: StateID,
        target_id: StateID,
        guard: Optional[AsyncGuard] = None,
        actions: Optional[List[AsyncAction]] = None,
        priority: int = 0,
    ):
        if not source_id or not target_id:
            raise ValueError("State IDs cannot be empty")
        self._source_id = source_id
        self._target_id = target_id
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state_id(self) -> StateID:
        return self._source_id

    def get_target_state_id(self) -> StateID:
        return self._target_id

    def get_guard(self) -> Optional[AsyncGuard]:
        return self._guard

    def get_actions(self) -> List[AsyncAction]:
        return self._actions

    def get_priority(self) -> int:
        return self._priority


class AsyncStateMachine(AsyncHSMBase):
    """Async variant of the state machine implementation."""

    def __init__(
        self,
        states: List[AsyncState],
        transitions: List[AsyncTransition],
        initial_state: AsyncState,
        max_queue_size: Optional[int] = None,
    ):
        """Initialize async state machine."""
        if not states or not transitions or not initial_state:
            raise ValueError("States, transitions, and initial state are required")
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")

        self._states = {state.get_id(): state for state in states}
        self._transitions = transitions
        self._initial_state = initial_state
        self._current_state: Optional[AsyncState] = None
        self._event_queue = AsyncEventQueue(max_size=max_queue_size)
        self._timer = AsyncTimer()
        self._running = False
        self._state_changes: Set[StateID] = set()
        self._state_change_callbacks: List[Callable[[StateID, StateID], None]] = []
        self._lock_manager = AsyncLockManager()

    async def start(self) -> None:
        """Start the state machine."""
        if self._running:
            return

        self._running = True
        try:
            self._current_state = self._initial_state
            await self._current_state.on_enter()
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
                await self._current_state.on_exit()
            await self._event_queue.shutdown()
            await self._timer.shutdown()
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
                await self._lock_manager.with_lock("processing", lambda e=event: self._handle_event(e))
            except Exception as e:
                logger.error("Error processing event: %s", str(e))
                if not isinstance(e, EventQueueError):
                    logger.exception("Unexpected error in event processing loop")

    async def _handle_event(self, event: AbstractEvent) -> None:
        """Handle a single event."""
        if not self._current_state:
            return

        async with self._lock_manager.with_lock("transition", lambda: self._find_valid_transition(event)) as transition:
            if transition:
                await self._execute_transition(transition, event)

    async def _find_valid_transition(self, event: AbstractEvent) -> Optional[AsyncTransition]:
        """Find the highest priority valid transition for the current event."""
        valid_transitions = []
        current_state_id = self._current_state.get_id()

        for transition in self._transitions:
            if transition.get_source_state_id() != current_state_id:
                continue

            guard = transition.get_guard()
            if guard:
                try:
                    if await guard.check(event, self._current_state.data):
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

        source_id = transition.get_source_state_id()
        target_id = transition.get_target_state_id()
        target_state = self._states.get(target_id)

        self._validate_transition(transition, self._states)

        try:
            await self._do_transition(self._current_state, target_state, transition, event)
        except Exception as e:
            raise self.wrap_error(
                e, "transition execution", source_id=source_id, target_id=target_id, event=event
            ) from e

    def get_current_state(self) -> Optional[AsyncState]:
        """Get the current state."""
        return self._current_state

    def is_running(self) -> bool:
        """Check if the state machine is running."""
        return self._running

    async def reset(self) -> None:
        """Reset the state machine to its initial state."""
        if self._running:
            await self.stop()

        self._current_state = None
        self._state_changes.clear()
        await self._event_queue.clear()
        await self.start()

    async def __aenter__(self) -> "AsyncStateMachine":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    def add_state_change_callback(self, callback: Callable[[StateID, StateID], None]) -> None:
        self._state_change_callbacks.append(callback)

    def validate(self) -> None:
        """Validate state machine configuration."""
        if not self._states:
            raise ValueError("No states defined")
        if not self._transitions:
            raise ValueError("No transitions defined")

        # Validate all transitions reference valid states
        for transition in self._transitions:
            if transition.get_source_state_id() not in self._states:
                raise ValueError(f"Invalid source state: {transition.get_source_state_id()}")
            if transition.get_target_state_id() not in self._states:
                raise ValueError(f"Invalid target state: {transition.get_target_state_id()}")

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the state machine."""
        return {
            "current_state": self._current_state.get_id() if self._current_state else None,
            "running": self._running,
            "state_changes": list(self._state_changes),
            "queue_size": self._event_queue.size(),
            "states": list(self._states.keys()),
            "transitions": [
                {"source": t.get_source_state_id(), "target": t.get_target_state_id(), "priority": t.get_priority()}
                for t in self._transitions
            ],
        }

    async def _handle_async_operation(
        self, operation: Callable[[], Awaitable[T]], error_class: Type[AsyncHSMError], error_msg: str, **error_kwargs
    ) -> T:
        """Helper for consistent error handling of async operations."""
        try:
            return await operation()
        except Exception as e:
            raise error_class(f"{error_msg}: {str(e)}", **error_kwargs) from e

    def requires_current_state(self, f: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[Optional[T]]]:
        """Decorator to ensure current state exists before operation."""

        @wraps(f)
        async def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            if not self._current_state:
                return None
            return await f(self, *args, **kwargs)

        return wrapper

    @contextmanager
    async def _protected_operation(self, lock: asyncio.Lock):
        """Context manager for lock-protected operations."""
        async with lock:
            yield

    def _notify_state_change(self, source_id: StateID, target_id: StateID) -> None:
        """Notify all callbacks of a state change."""
        for callback in self._state_change_callbacks:
            callback(source_id, target_id)

    async def _do_transition(
        self, current_state: AsyncState, target_state: AsyncState, transition: AsyncTransition, event: AbstractEvent
    ) -> None:
        """Execute transition steps."""
        await current_state.on_exit()

        for action in transition.get_actions():
            await action.execute(event, current_state.data)

        self._current_state = target_state
        self._state_changes.add(target_state.get_id())

        await target_state.on_enter()

        self._notify_state_change(current_state.get_id(), target_state.get_id())
