# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hsm.core.events import Event
    from hsm.core.hooks import HookProtocol
    from hsm.core.states import State
    from hsm.core.validations import Validator


class AsyncStateMachine:
    """
    An asynchronous version of the state machine. Allows event processing in an
    async context, integrating with asyncio-based loops and async event queues.
    """

    def __init__(
        self, initial_state: "State", validator: "Validator" = None, hooks: list["HookProtocol"] = None
    ) -> None:
        """
        Initialize with an initial state, optional validator, and hooks, similar
        to the synchronous StateMachine but for async usage.
        """

    async def start(self) -> None:
        """
        Start the async state machine, performing validation and initializing the state.
        """

    async def process_event(self, event: "Event") -> None:
        """
        Asynchronously process an event, performing transitions if needed.

        :param event: The event to handle.
        """

    async def stop(self) -> None:
        """
        Asynchronously stop the state machine, allowing final cleanup actions.
        """

    @property
    def current_state(self) -> "State":
        """
        Return the current state in an async-compatible manner.
        """


class AsyncEventQueue:
    """
    An asynchronous event queue providing non-blocking enqueue/dequeue methods,
    suitable for use with AsyncStateMachine.
    """

    def __init__(self, priority: bool = False) -> None:
        """
        Initialize the async event queue.

        :param priority: If True, operates in a priority-based mode.
        """
        raise NotImplementedError()

    async def enqueue(self, event: "Event") -> None:
        """
        Asynchronously insert an event into the queue.
        """
        raise NotImplementedError()

    async def dequeue(self) -> "Event" | None:
        """
        Asynchronously retrieve the next event, or None if empty.
        """
        raise NotImplementedError()

    async def clear(self) -> None:
        """
        Asynchronously clear all events from the queue.
        """
        raise NotImplementedError()

    @property
    def priority_mode(self) -> bool:
        """
        Indicates if this async queue uses priority ordering.
        """
        raise NotImplementedError()


class _AsyncEventProcessingLoop:
    """
    Internal async loop for event processing, integrating with asyncio's event loop
    to continuously process events until stopped.
    """

    def __init__(self, machine: "AsyncStateMachine", event_queue: "AsyncEventQueue") -> None:
        """
        Store references for async iteration.
        """
        raise NotImplementedError()

    async def start_loop(self) -> None:
        """
        Begin processing events asynchronously.
        """
        raise NotImplementedError()

    async def stop_loop(self) -> None:
        """
        Stop processing events, allowing async tasks to conclude gracefully.
        """
        raise NotImplementedError()


class _AsyncLock:
    """
    Internal async-compatible lock abstraction, providing awaitable acquisition
    methods.
    """

    def __init__(self) -> None:
        """
        Initialize internal async lock.
        """
        raise NotImplementedError()

    async def acquire(self) -> None:
        """
        Await the lock until it is acquired.
        """
        raise NotImplementedError()

    def release(self) -> None:
        """
        Release the async lock.
        """
        raise NotImplementedError()
