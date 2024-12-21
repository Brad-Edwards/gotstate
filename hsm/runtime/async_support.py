# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import asyncio
from typing import List, Optional

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.hooks import HookManager, HookProtocol
from hsm.core.state_machine import StateMachine, _StateMachineContext
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

    def __init__(self, timeout: float = 0.1):
        self._queue = asyncio.PriorityQueue()
        self._timeout = timeout

    async def enqueue(self, event: Event) -> None:
        """Add event to queue using event's priority."""
        # Event already has priority, no need to tuple
        await self._queue.put(event)

    async def dequeue(self) -> Optional[Event]:
        """Get next event respecting priority."""
        try:
            event = await asyncio.wait_for(self._queue.get(), timeout=self._timeout)
            self._queue.task_done()
            return event
        except asyncio.TimeoutError:
            return None

    async def clear(self) -> None:
        """Clear all pending events."""
        while not self._queue.empty():
            try:
                await self.dequeue()
            except asyncio.TimeoutError:
                break


class AsyncStateMachine(StateMachine):
    """
    Asynchronous version of the state machine that supports async event processing.
    """

    def __init__(self, initial_state: State, validator: Optional[Validator] = None, hooks: Optional[List] = None):
        super().__init__(initial_state, validator, hooks)
        self._event_queue = AsyncEventQueue()
        self._async_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the state machine with async validation."""
        async with self._async_lock:
            if self._started:
                return

            # Reuse parent's resolution logic
            self._current_state = self._resolve_state_for_start()

            # Validate using parent's validation but handle async
            errors = self._graph.validate()
            if errors:
                raise ValidationError("\n".join(errors))

            if asyncio.iscoroutinefunction(self._validator.validate_state_machine):
                await self._validator.validate_state_machine(self)
            else:
                self._validator.validate_state_machine(self)

            await self._notify_enter_async(self._current_state)
            self._started = True

    async def stop(self) -> None:
        """Stop the state machine asynchronously."""
        if not self._started:
            return
        if self._current_state:
            await self._notify_exit_async(self._current_state)
        self._current_state = None
        self._started = False

    async def process_event(self, event: Event) -> bool:
        """Process an event asynchronously."""
        async with self._async_lock:
            # Reuse parent's validation and transition selection
            if not self._started or not self._current_state:
                return False

            valid_transitions = self._graph.get_valid_transitions(self._current_state, event)
            if not valid_transitions:
                return False

            transition = max(valid_transitions, key=lambda t: t.get_priority())
            await self._execute_transition_async(transition, event)
            return True

    async def _execute_transition_async(self, transition: Transition, event: Event) -> None:
        """Execute transition actions asynchronously while reusing parent's logic."""
        try:
            await self._notify_exit_async(self._current_state)

            # Execute actions with async support
            for action in transition.actions:
                try:
                    if asyncio.iscoroutinefunction(action):
                        await action(event)
                    else:
                        action(event)
                except Exception as e:
                    await self._notify_error_async(e)
                    return

            # Reuse parent's state update logic
            self._current_state = transition.target
            await self._notify_enter_async(self._current_state)

        except Exception as e:
            await self._notify_error_async(e)

    async def _notify_enter_async(self, state: State) -> None:
        """Notify state entry asynchronously."""
        # Call state's own enter method first
        state.on_enter()
        # Then notify hooks
        for hook in self._hooks:
            if hasattr(hook, "on_enter"):
                hook_method = hook.on_enter
                if asyncio.iscoroutinefunction(hook_method):
                    await hook_method(state)
                else:
                    hook_method(state)

    async def _notify_exit_async(self, state: State) -> None:
        """Notify hooks of state exit asynchronously."""
        # No try/except here - let errors propagate up
        state.on_exit()
        for hook in self._hooks:
            if hasattr(hook, "on_exit"):
                hook_method = hook.on_exit
                if asyncio.iscoroutinefunction(hook_method):
                    await hook_method(state)
                else:
                    hook_method(state)

    async def _notify_error_async(self, error: Exception) -> None:
        """Notify hooks of an error asynchronously."""
        for hook in self._hooks:
            if hasattr(hook, "on_error"):
                hook_method = hook.on_error
                if asyncio.iscoroutinefunction(hook_method):
                    await hook_method(error)
                else:
                    hook_method(error)


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
