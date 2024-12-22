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
from hsm.core.states import CompositeState, State
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
        # Don't set current state until start
        self._graph.set_current_state(None)

    @property
    def current_state(self) -> Optional[State]:
        """Get the current state from the graph."""
        return self._graph.get_current_state()

    def _set_current_state(self, state: Optional[State]) -> None:
        """Set the current state in the graph."""
        self._graph.set_current_state(state)

    async def start(self) -> None:
        """Start the state machine with async validation."""
        async with self._async_lock:
            if self._started:
                return

            # Validate before entering state
            errors = self._graph.validate()
            if errors:
                raise ValidationError("\n".join(errors))

            # Validator may be async or sync
            if asyncio.iscoroutinefunction(self._validator.validate_state_machine):
                await self._validator.validate_state_machine(self)
            else:
                self._validator.validate_state_machine(self)

            # Resolve initial or historical active state
            initial_state = self._graph.get_initial_state(None)  # Get root initial state
            if initial_state:
                resolved_state = self._graph.resolve_active_state(initial_state)
                self._set_current_state(resolved_state)
                # Now enter the state
                await self._notify_enter_async(self.current_state)

            self._started = True

    async def stop(self) -> None:
        """Stop the state machine asynchronously."""
        async with self._async_lock:
            if not self._started:
                return

            if self.current_state:
                await self._notify_exit_async(self.current_state)
                self._set_current_state(None)

            self._started = False

    async def process_event(self, event: Event) -> bool:
        """Process an event asynchronously."""
        if not self._started:
            return False

        async with self._async_lock:
            valid_transitions = []
            if self.current_state:  # Only process transitions if we have a current state
                # First try transitions from current state
                for transition in self._graph.get_valid_transitions(self.current_state, event):
                    if await transition.evaluate_guards(event):
                        valid_transitions.append(transition)

                # If no transitions found, try parent states
                if not valid_transitions:
                    current = self.current_state.parent
                    while current and not valid_transitions:
                        for transition in self._graph.get_valid_transitions(current, event):
                            if await transition.evaluate_guards(event):
                                valid_transitions.append(transition)
                        current = current.parent

            if not valid_transitions:
                return False

            # Pick highest-priority transition
            transition = max(valid_transitions, key=lambda t: t.get_priority())
            result = await self._execute_transition_async(transition, event)
            # If result is False, transition failed but was handled
            return result if result is not None else True

    async def _execute_transition_async(self, transition: Transition, event: Event) -> None:
        """Execute a transition asynchronously."""
        previous_state = self.current_state
        try:
            # Find the common ancestor between source and target states
            source_ancestors = []
            current = self.current_state
            while current:
                source_ancestors.append(current)
                current = current.parent

            target_ancestors = []
            current = transition.target
            while current:
                target_ancestors.append(current)
                current = current.parent

            # Find common ancestor
            common_ancestor = None
            for state in source_ancestors:
                if state in target_ancestors:
                    common_ancestor = state
                    break

            # Exit up to common ancestor
            current = self.current_state
            while current and current != common_ancestor:
                await self._notify_exit_async(current)
                current = current.parent

            # Execute transition actions
            for action in transition.actions:
                if asyncio.iscoroutinefunction(action):
                    await action(event)
                else:
                    action(event)

            # Update current state
            self._set_current_state(transition.target)

            # Notify transition
            for hook in self._hooks:
                if hasattr(hook, "on_transition"):
                    if asyncio.iscoroutinefunction(hook.on_transition):
                        await hook.on_transition(transition.source, transition.target)
                    else:
                        hook.on_transition(transition.source, transition.target)

            # Enter from common ancestor to target
            target_path = []
            current = transition.target
            while current and current != common_ancestor:
                target_path.append(current)
                current = current.parent

            for state in reversed(target_path):
                await self._notify_enter_async(state)

        except Exception as e:
            # Restore previous state if we failed during transition
            self._set_current_state(previous_state)
            await self._notify_error_async(e)
            # Don't re-raise the exception since we've handled it
            return False

    async def _notify_enter_async(self, state: State) -> None:
        """Invoke on_enter hooks asynchronously."""
        if asyncio.iscoroutinefunction(state.on_enter):
            await state.on_enter()
        else:
            state.on_enter()

        for hook in self._hooks:
            if hasattr(hook, "on_enter"):
                if asyncio.iscoroutinefunction(hook.on_enter):
                    await hook.on_enter(state)
                else:
                    hook.on_enter(state)

    async def _notify_exit_async(self, state: State) -> None:
        """Invoke on_exit hooks asynchronously."""
        if asyncio.iscoroutinefunction(state.on_exit):
            await state.on_exit()
        else:
            state.on_exit()

        for hook in self._hooks:
            if hasattr(hook, "on_exit"):
                if asyncio.iscoroutinefunction(hook.on_exit):
                    await hook.on_exit(state)
                else:
                    hook.on_exit(state)

    async def _notify_error_async(self, error: Exception) -> None:
        """Invoke on_error hooks asynchronously."""
        for hook in self._hooks:
            if hasattr(hook, "on_error"):
                if asyncio.iscoroutinefunction(hook.on_error):
                    await hook.on_error(error)
                else:
                    hook.on_error(error)


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


class AsyncCompositeStateMachine(AsyncStateMachine):
    """
    Asynchronous version of CompositeStateMachine that properly handles
    submachine transitions with async locking.
    """

    def __init__(
        self,
        initial_state: State,
        validator: Optional[Validator] = None,
        hooks: Optional[List] = None,
    ):
        super().__init__(initial_state, validator, hooks)
        self._submachines = {}

    def add_submachine(self, state: CompositeState, submachine: "AsyncStateMachine") -> None:
        """
        Add a submachine's states under a parent composite state.
        Submachine's states are all integrated into this machine's graph.
        """
        if not isinstance(state, CompositeState):
            raise ValueError(f"State {state.name} must be a composite state")

        # First add the composite state if it's not already in the graph
        if state not in self._graph._nodes:
            self._graph.add_state(state)

        # Integrate submachine states into the same graph:
        for sub_state in submachine.get_states():
            # Add each state with the composite state as parent
            self._graph.add_state(sub_state, parent=state)

        # Integrate transitions
        for t in submachine.get_transitions():
            self._graph.add_transition(t)

        # Set the composite state's initial state to the submachine's initial state
        submachine_initial = submachine._graph.get_initial_state(None)  # Get root initial state
        if submachine_initial:
            # Set initial state in the graph's internal state
            self._graph.set_initial_state(state, submachine_initial)
            # Add a transition from the composite state to its initial state
            self._graph.add_transition(Transition(source=state, target=submachine_initial, guards=[lambda e: True]))

        self._submachines[state] = submachine

    async def process_event(self, event: Event) -> bool:
        """
        Process events with proper handling of submachine hierarchy and async locking.
        Submachine transitions take precedence over parent transitions.
        """
        if not self._started or not self.current_state:
            return False

        async with self._async_lock:  # Ensure thread-safe state access
            try:
                # First try transitions from the current state
                current_transitions = self._graph.get_valid_transitions(self.current_state, event)

                # Then check transitions from ancestors
                current = self.current_state.parent
                while current:
                    transitions = self._graph.get_valid_transitions(current, event)
                    current_transitions.extend(transitions)
                    current = current.parent

                if not current_transitions:
                    return False

                # Sort transitions by priority
                current_transitions.sort(key=lambda t: t.get_priority(), reverse=True)

                # Evaluate guards to find valid transitions
                valid_transitions = []
                for transition in current_transitions:
                    # Check if all guards pass
                    all_guards_pass = True
                    for guard in transition.guards:
                        if asyncio.iscoroutinefunction(guard):
                            if not await guard(event):
                                all_guards_pass = False
                                break
                        elif not guard(event):
                            all_guards_pass = False
                            break
                    if all_guards_pass:
                        valid_transitions.append(transition)
                        break  # Take the first valid transition, respecting priority order

                if not valid_transitions:
                    return False

                # Execute the highest priority valid transition
                transition = valid_transitions[0]
                result = await self._execute_transition_async(transition, event)

                # If target is a composite state, enter its initial state
                if isinstance(transition.target, CompositeState):
                    initial_state = self._graph.get_initial_state(transition.target)
                    if initial_state:
                        # Create and execute a transition to the initial state
                        initial_transition = Transition(
                            source=transition.target, target=initial_state, guards=[lambda e: True]
                        )
                        await self._execute_transition_async(initial_transition, event)

                return result if result is not None else True

            except Exception as error:
                await self._notify_error_async(error)
                raise
