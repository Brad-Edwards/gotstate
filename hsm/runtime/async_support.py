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
    AsyncGenerator,
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
    cast,
    runtime_checkable,
)

from hsm.core.errors import ExecutorError, HSMError, InvalidStateError
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

EXECUTOR_IS_NOT_RUNNING = "Executor is not running"
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

    def get_locks(self) -> Dict[str, asyncio.Lock]:
        """Get the current locks dictionary."""
        return self._locks


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
            except AsyncStateError as e:
                raise e
            except Exception as e:
                if hasattr(e, "__dict__"):
                    details = e.__dict__
                else:
                    details = {"error": str(e)}
                raise AsyncStateError(f"Error during state entry: {str(e)}", self._state_id, details) from e

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

        if not substates:
            raise ValueError("Composite state must have at least one substate")

        if initial_state is None:
            initial_state = substates[0]
        elif initial_state not in substates:
            raise ValueError("initial_state must be one of the substates")

        self._substates = substates
        self._initial_state = initial_state
        self._has_history = has_history
        self._history_state: Optional[AbstractState] = None
        self._current_substate = initial_state
        self._parent_state = parent_state

        # Set parent references for all substates
        for substate in substates:
            if isinstance(substate, AsyncCompositeState):
                substate.parent_state = self

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
            if self.has_history():
                self.set_history_state(self._current_substate)
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

    def set_has_history(self, value: bool) -> None:
        """Set whether this state maintains history."""
        self._has_history = value

    def get_history_state(self) -> Optional[AbstractState]:
        """Get the current history state."""
        return self._history_state


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

    def __init__(self, states: List[AsyncState], transitions: List[AsyncTransition], initial_state: AsyncState):
        """Initialize async state machine."""
        if not states:
            raise ValueError("States list cannot be empty")
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")

        self._states: Dict[StateID, AsyncState] = {state.get_id(): state for state in states}
        self._transitions: List[AsyncTransition] = transitions
        self._initial_state: AsyncState = initial_state
        self._current_state: Optional[AsyncState] = None
        self._state_data: Dict[StateID, Any] = {}
        self._initialize_state_data()

        # Initialize timer with default callback
        async def timer_callback(timer_id: str, event: AbstractEvent) -> None:
            try:
                await self.process_event(event)
            except Exception as e:
                logger.error("Timer callback failed: %s", str(e))
                # You might want to handle this error more gracefully, e.g., retry or set an error state
                # self._context.handle_error(e)

        self._timer: AsyncTimer = AsyncTimer("state_machine_timer", timer_callback)
        self._running: bool = False
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
                await self._exit_states(
                    self._current_state, self._current_state, None, self._get_state_data(self._current_state.get_id())
                )
            await self._timer.shutdown()
            # Wait for event processing loop to complete
            await asyncio.sleep(0)  # Give the event loop a chance to finish
        except Exception as e:
            logger.error("Error during state machine shutdown: %s", str(e))
            raise AsyncHSMError(f"Failed to stop state machine: {str(e)}") from e

    async def _handle_event(self, event: AbstractEvent) -> None:
        """Handle a single event."""
        logger.debug("Starting event handling")
        if not self._current_state:
            logger.debug("No current state, skipping event")
            return

        try:
            logger.debug(f"Finding valid transition from state: {self._current_state.get_id()}")

            # Get the transition first
            async def find_transition():
                return await self._find_valid_transition(event)

            transition = await self._lock_manager.with_lock("transition", find_transition)

            if transition:
                logger.debug(f"Found valid transition to: {transition.get_target_state().get_id()}")
                await self._execute_transition(transition, event)
                logger.debug("Transition executed successfully")
            else:
                logger.debug("No valid transition found for event")

        except Exception as e:
            logger.error(f"Error handling event: {str(e)}")
            raise
        finally:
            logger.debug("Event handling complete")

    async def _find_valid_transition(self, event: AbstractEvent) -> Optional[AsyncTransition]:
        """Find a valid transition for the current state and event."""
        if not self._current_state:
            return None

        for transition in self._transitions:
            if transition.get_source_state() == self._current_state:
                try:
                    guard = transition.get_guard()
                    if guard:
                        if not await guard.check(event, self._get_state_data(self._current_state.get_id())):
                            continue
                    return transition
                except Exception as e:
                    logger.error(f"Guard check failed: {str(e)}")
                    raise AsyncTransitionError(
                        f"Guard check failed: {str(e)}",
                        transition.get_source_state(),
                        transition.get_target_state(),
                        event,
                        {"error": str(e)},
                    ) from e
        return None

    async def _execute_transition(self, transition: AsyncTransition, event: AbstractEvent) -> None:
        """Execute a transition."""
        if not self._current_state:
            logger.debug("No current state, skipping transition")
            return

        source_state = transition.get_source_state()
        target_state = transition.get_target_state()
        source_data = self._get_state_data(source_state.get_id())
        target_data = self._get_state_data(target_state.get_id())

        logger.debug(f"Starting transition from {source_state.get_id()} to {target_state.get_id()}")

        try:
            # Exit states
            logger.debug("Exiting states")
            await self._exit_states(source_state, target_state, event, source_data)

            # Execute transition actions sequentially
            logger.debug("Executing transition actions")
            for action in transition.get_actions():
                await action.execute(event, source_data)

            # Update current state before entering new states
            logger.debug("Updating current state")
            self._current_state = target_state

            # Enter states
            logger.debug("Entering new states")
            await self._enter_states(source_state, target_state, event, target_data)

            logger.debug(f"Transition complete. Current state: {self._current_state.get_id()}")

        except Exception as e:
            logger.error(f"Error during transition: {str(e)}")
            # Restore original state on error
            self._current_state = source_state
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
            is_target_substate = False
            if source_state.get_substates():
                for substate in source_state.get_substates():
                    if substate == target_state:
                        is_target_substate = True
                        break
            if source_state._current_substate:
                await source_state._current_substate.on_exit(event, data)
                if not is_target_substate:
                    source_state._current_substate = None
            await source_state.on_exit(event, data)
            if not is_target_substate:
                self._current_state = None

    async def _enter_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """Helper function for entering states during a transition."""
        try:
            # Build a path of states to enter from the source state to the target state
            path_to_target = []
            current_state = target_state
            max_depth = 100  # Prevent infinite loops
            depth = 0

            while current_state != source_state and depth < max_depth:
                path_to_target.insert(0, current_state)
                if isinstance(current_state, AsyncCompositeState) and current_state._parent_state:
                    current_state = current_state._parent_state
                else:
                    break
                depth += 1

            if depth >= max_depth:
                raise AsyncStateError("Maximum state nesting depth exceeded", target_state.get_id())

            # Enter all states in the path
            for state in path_to_target:
                await state.on_entry(event, self._get_state_data(state.get_id()))
                if isinstance(state, AsyncCompositeState):
                    await state._enter_substate(event, data)
                    if state.has_history():
                        state.set_history_state(state._current_substate)

            self._current_state = target_state

        except Exception as e:
            logger.error(f"Error entering states: {str(e)}")
            raise AsyncStateError(f"Failed to enter states: {str(e)}", target_state.get_id()) from e

    def get_state(self) -> Optional[AsyncState]:
        """Get the current state."""
        return self._current_state

    def is_running(self) -> bool:
        """Check if the state machine is running."""
        return self._running

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


class AsyncExecutor:
    """
    Asynchronous event processor for the async state machine.
    """

    def __init__(self, state_machine: AsyncStateMachine, max_queue_size: Optional[int] = None):
        if not state_machine:
            raise ValueError("state_machine cannot be None")
        self._state_machine = state_machine
        self._event_queue: AsyncEventQueue = AsyncEventQueue(max_size=max_queue_size)
        self._task: Optional[asyncio.Task] = None
        self._processing_flag: asyncio.Event = asyncio.Event()
        self._processing_flag.set()

        # Initialize timer with default callback
        async def timer_callback(timer_id: str, event: AbstractEvent) -> None:
            try:
                if self._processing_flag.is_set():
                    await self.process_event(event)
            except Exception as e:
                logger.error("Timer callback failed: %s", str(e))
                # You might want to handle this error more gracefully, e.g., retry or set an error state
                # self._context.handle_error(e)

        self._timer: AsyncTimer = AsyncTimer("state_machine_timer", timer_callback)

    async def start(self) -> None:
        """Start the executor and the state machine."""
        if self._task is not None:
            raise ExecutorError("Executor already started")

        await self._state_machine.start()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        """Stop the executor and the state machine."""
        if self._task is None:
            return

        try:
            await self._state_machine.stop()
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling the task
        finally:
            self._task = None

    async def process_event(self, event: AbstractEvent) -> None:
        """Enqueue an event for processing."""
        if self._task is None:
            raise ExecutorError("Executor is not running")
        await self._event_queue.enqueue(event)

    async def _run_forever(self) -> None:
        """Main event processing loop."""
        while True:
            try:
                if not self._processing_flag.is_set():
                    await self._processing_flag.wait()  # Pause until flag is set

                event = await self._event_queue.try_dequeue(timeout=0.1)
                if event:
                    try:
                        await self._state_machine._handle_event(event)
                    except AsyncHSMError as e:
                        logger.error("State machine error processing event: %s", str(e))
                        # Continue processing next events but log the error
                        continue
                    except Exception as e:
                        logger.exception("Unexpected error processing event: %s", str(e))
                        # For unexpected errors, wait a bit before continuing to prevent tight error loops
                        await asyncio.sleep(0.1)
                        continue

            except asyncio.CancelledError:
                logger.debug("Event processing loop cancelled")
                break  # Exit loop on cancellation
            except EventQueueError as e:
                logger.error("Event queue error: %s", str(e))
                # Wait a bit before retrying queue operations
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.exception("Critical error in event processing loop: %s", str(e))
                # Wait longer for critical errors before retrying
                await asyncio.sleep(1.0)

    @asynccontextmanager
    async def pause(self) -> AsyncGenerator[None, None]:
        """Pause event processing."""
        if not self.is_running():
            raise ExecutorError(EXECUTOR_IS_NOT_RUNNING)

        self._processing_flag.clear()  # Clear the flag to pause
        try:
            yield
        finally:
            self._processing_flag.set()

    def resume(self) -> None:
        """Explicitly resumes event processing (can be called outside of the pause context)."""
        if not self.is_running():
            raise ExecutorError("Executor is not running")

        self._processing_flag.set()

    def is_running(self) -> bool:
        """Check if the executor is running."""
        return self._task is not None and not self._task.done()
