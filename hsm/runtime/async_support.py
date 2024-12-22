# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import asyncio
from typing import List, Optional

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import StateMachine  # Removed _StateMachineContext import
from hsm.core.states import State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator


class _AsyncLock:
    """
    Internal async-compatible lock abstraction, providing awaitable acquisition
    methods. Only keep if actually needed; otherwise, you can remove.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._lock.acquire()

    def release(self) -> None:
        self._lock.release()


class AsyncEventQueue:
    """
    Asynchronous event queue implementation supporting priority-based ordering.
    """

    def __init__(self, priority: bool = True):
        """
        Initialize async event queue.

        :param priority: If True, enables priority-based event processing.
                         If False, uses standard FIFO ordering.
        """
        self.priority_mode = priority
        self._queue = asyncio.PriorityQueue() if priority else asyncio.Queue()
        self._running = True
        self._counter = 0

    async def enqueue(self, event: Event) -> None:
        """Add an event to the queue."""
        if self.priority_mode:
            # Negate event.priority so higher event.priority => higher priority => dequeued sooner
            await self._queue.put((-event.priority, self._counter, event))
            self._counter += 1
        else:
            await self._queue.put(event)

    async def dequeue(self) -> Optional[Event]:
        """
        Remove and return the next event from the queue.
        Returns None if queue is empty after timeout or if the queue is stopped.
        """
        if not self._running:
            return None

        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            if self.priority_mode:
                return item[2]  # Return the Event from the tuple
            return item
        except asyncio.TimeoutError:
            return None

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    async def clear(self) -> None:
        """Clear all events from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stop(self) -> None:
        """Stop the queue processing."""
        self._running = False
        await self.clear()


class AsyncStateMachine(StateMachine):
    """
    Asynchronous version of the state machine that supports async event processing.
    """

    def __init__(self, initial_state: State, validator: Optional[Validator] = None, hooks: Optional[List] = None):
        super().__init__(initial_state, validator, hooks)
        self._async_lock = asyncio.Lock()
        self._current_action: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the state machine with async validation."""
        async with self._async_lock:
            if self._started:
                return

            self._current_state = self._graph.resolve_active_state(self._initial_state)

            errors = self._graph.validate()
            if errors:
                raise ValidationError("\n".join(errors))

            # Validator may be async or sync
            if asyncio.iscoroutinefunction(self._validator.validate_state_machine):
                await self._validator.validate_state_machine(self)
            else:
                self._validator.validate_state_machine(self)

            await self._notify_enter_async(self._current_state)
            self._started = True

    async def stop(self) -> None:
        """Stop the state machine asynchronously."""
        async with self._async_lock:
            if not self._started:
                return

            if self._current_state:
                await self._notify_exit_async(self._current_state)

            self._current_state = None
            self._started = False

    async def process_event(self, event: Event) -> bool:
        """Process an event asynchronously with proper cancellation support."""
        if not self._started or not self._current_state:
            return False

        try:
            async with self._async_lock:
                valid_transitions = self._graph.get_valid_transitions(self._current_state, event)
                if not valid_transitions:
                    return False

                transition = max(valid_transitions, key=lambda t: t.get_priority())
                return await self._execute_transition_async(transition, event)

        except asyncio.CancelledError:
            # If we get cancelled while holding the lock, ensure cleanup
            if self._current_action and not self._current_action.done():
                self._current_action.cancel()
                try:
                    await asyncio.shield(self._current_action)
                except (asyncio.CancelledError, Exception):
                    pass
            raise

    async def _execute_transition_async(self, transition: Transition, event: Event) -> bool:
        """Execute transition actions with proper cancellation support."""
        previous_state = self._current_state
        try:
            await self._notify_exit_async(self._current_state)

            # Execute actions (some might be async)
            for action in transition.actions:
                if asyncio.iscoroutinefunction(action):
                    try:
                        self._current_action = asyncio.create_task(action(event))
                        await self._current_action
                    except asyncio.CancelledError:
                        await self._rollback_transition(previous_state)
                        raise
                    except Exception as e:
                        await self._rollback_transition(previous_state)
                        await self._notify_error_async(e)
                        return False
                    finally:
                        self._current_action = None
                else:
                    try:
                        action(event)
                    except Exception as e:
                        await self._rollback_transition(previous_state)
                        await self._notify_error_async(e)
                        return False

            # Update state
            self._current_state = transition.target
            # Notify hooks of the transition
            await self._notify_transition_async(previous_state, self._current_state)
            await self._notify_enter_async(self._current_state)
            return True

        except asyncio.CancelledError:
            await self._rollback_transition(previous_state)
            raise
        except Exception as e:
            await self._rollback_transition(previous_state)
            await self._notify_error_async(e)
            return False

    async def _rollback_transition(self, previous_state: State) -> None:
        """Roll back to previous state during cancellation or error."""
        self._current_state = previous_state
        await self._notify_enter_async(previous_state)

    async def _notify_enter_async(self, state: State) -> None:
        """Notify state entry asynchronously."""
        state.on_enter()
        for hook in self._hooks:
            if hasattr(hook, "on_enter"):
                hook_method = hook.on_enter
                if asyncio.iscoroutinefunction(hook_method):
                    await hook_method(state)
                else:
                    hook_method(state)

    async def _notify_exit_async(self, state: State) -> None:
        """Notify hooks of state exit asynchronously."""
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

    async def _notify_transition_async(self, source: State, target: State) -> None:
        """Notify hooks of a state transition asynchronously."""
        for hook in self._hooks:
            if hasattr(hook, "on_transition"):
                hook_method = hook.on_transition
                if asyncio.iscoroutinefunction(hook_method):
                    await hook_method(source, target)
                else:
                    hook_method(source, target)


class _AsyncEventProcessingLoop:
    """
    Internal async loop for event processing, integrating with asyncio's event loop
    to continuously process events until stopped.
    """

    def __init__(self, machine: AsyncStateMachine, event_queue: AsyncEventQueue) -> None:
        self._machine = machine
        self._queue = event_queue
        self._running = False

    async def start_loop(self) -> None:
        """Begin processing events asynchronously."""
        self._running = True
        await self._machine.start()  # Ensure machine is started

        while self._running:
            event = await self._queue.dequeue()
            if event:
                await self._machine.process_event(event)
            else:
                await asyncio.sleep(0.01)

    async def stop_loop(self) -> None:
        """Stop processing events, letting async tasks conclude gracefully."""
        self._running = False
        await self._machine.stop()


def create_nested_state_machine(hook) -> AsyncStateMachine:
    """Create a nested state machine for testing."""
    root = State("Root")
    processing = State("Processing")
    error = State("Error")
    operational = State("Operational")
    shutdown = State("Shutdown")

    machine = AsyncStateMachine(initial_state=root, hooks=[hook])

    machine.add_state(processing)
    machine.add_state(error)
    machine.add_state(operational)
    machine.add_state(shutdown)

    machine.add_transition(Transition(source=root, target=processing, guards=[lambda e: e.name == "begin"]))
    machine.add_transition(Transition(source=processing, target=operational, guards=[lambda e: e.name == "complete"]))
    machine.add_transition(Transition(source=operational, target=processing, guards=[lambda e: e.name == "begin"]))
    machine.add_transition(Transition(source=processing, target=error, guards=[lambda e: e.name == "error"]))
    machine.add_transition(Transition(source=error, target=operational, guards=[lambda e: e.name == "recover"]))

    # High-priority shutdown from any state
    for st in [root, processing, error, operational]:
        machine.add_transition(
            Transition(
                source=st,
                target=shutdown,
                guards=[lambda e: e.name == "shutdown"],
                priority=10,
            )
        )

    return machine
