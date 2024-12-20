# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import asyncio
from typing import Optional

from hsm.core.events import Event
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator


class _AsyncLock:
    """
    Internal async-compatible lock abstraction, providing awaitable acquisition
    methods.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._lock.acquire()

    def release(self) -> None:
        self._lock.release()


class AsyncEventQueue:
    """
    An asynchronous event queue providing non-blocking enqueue/dequeue methods,
    suitable for use with AsyncStateMachine.
    """

    def __init__(self, priority: bool = False) -> None:
        """
        Initialize the async event queue.

        :param priority: If True, operates in a priority-based mode.
                         Currently, we only implement a simple FIFO using asyncio.Queue.
                         Priority mode could be implemented separately if needed.
        """
        self._priority_mode = priority
        # For simplicity, we ignore priority in the async variant and just use FIFO.
        self._queue = asyncio.Queue()

    async def enqueue(self, event: Event) -> None:
        """
        Asynchronously insert an event into the queue.
        """
        await self._queue.put(event)

    async def dequeue(self) -> Optional[Event]:
        """
        Asynchronously retrieve the next event, or None if empty.
        This will block until an event is available.
        If a non-blocking or timeout approach is needed, adapt accordingly.
        """
        if self._queue.empty():
            return None
        return await self._queue.get()

    async def clear(self) -> None:
        """
        Asynchronously clear all events from the queue.
        This is not a standard asyncio.Queue operation; we implement by draining.
        """
        while not self._queue.empty():
            await self._queue.get()

    @property
    def priority_mode(self) -> bool:
        """
        Indicates if this async queue uses priority ordering.
        """
        return self._priority_mode


class AsyncStateMachine:
    """
    An asynchronous version of the state machine. Allows event processing in an
    async context, integrating with asyncio-based loops and async event queues.

    This class parallels StateMachine, but provides async start/stop/process_event.
    """

    def __init__(self, initial_state: "State", validator: Validator = None, hooks: list["HookProtocol"] = None):
        self._sync_machine = StateMachine(initial_state=initial_state, validator=validator, hooks=hooks)
        self._lock = _AsyncLock()  # Protects machine operations in async context
        self._started = False
        self._stopped = False

    @property
    def current_state(self) -> "State":
        return self._sync_machine.current_state

    async def start(self) -> None:
        if self._started:
            return
        await self._lock.acquire()
        try:
            self._sync_machine.start()
            self._started = True
        finally:
            self._lock.release()

    async def process_event(self, event: Event) -> None:
        if not self._started or self._stopped:
            return
        await self._lock.acquire()
        try:
            self._sync_machine.process_event(event)
        finally:
            self._lock.release()

    async def stop(self) -> None:
        if self._stopped:
            return
        await self._lock.acquire()
        try:
            self._sync_machine.stop()
            self._stopped = True
        finally:
            self._lock.release()

    def add_transition(self, transition: "Transition") -> None:
        # Adding transitions might be synchronous. If dynamic addition at runtime is needed,
        # consider making this async as well. For now, we assume it's done before start.
        self._sync_machine.add_transition(transition)


class _AsyncEventProcessingLoop:
    """
    Internal async loop for event processing, integrating with asyncio's event loop
    to continuously process events until stopped.
    """

    def __init__(self, machine: AsyncStateMachine, event_queue: AsyncEventQueue) -> None:
        """
        Store references for async iteration.
        """
        self._machine = machine
        self._queue = event_queue
        self._running = False

    async def start_loop(self) -> None:
        """
        Begin processing events asynchronously.
        """
        self._running = True
        await self._machine.start()  # Ensure machine started

        while self._running:
            event = await self._queue.dequeue()
            if event:
                await self._machine.process_event(event)
            else:
                # If no event, we can await a small sleep or wait again
                await asyncio.sleep(0.01)

    async def stop_loop(self) -> None:
        """
        Stop processing events, allowing async tasks to conclude gracefully.
        """
        self._running = False
        await self._machine.stop()
