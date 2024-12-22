# hsm/runtime/async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Set

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.hooks import HookProtocol
from hsm.core.state_machine import StateMachine
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator


class AsyncStateHistory:
    """Thread-safe state history management for async state machines."""

    def __init__(self):
        self._history: Dict[CompositeState, State] = {}
        self._lock = asyncio.Lock()

    async def record_state(self, composite_state: CompositeState, state: State) -> None:
        """Thread-safe recording of state history."""
        async with self._lock:
            self._history[composite_state] = state

    async def get_last_state(self, composite_state: CompositeState) -> Optional[State]:
        """Thread-safe retrieval of last active state."""
        async with self._lock:
            return self._history.get(composite_state)

    async def clear(self) -> None:
        """Thread-safe clearing of history."""
        async with self._lock:
            self._history.clear()


class AsyncStateGraph:
    """Thread-safe state graph for async state machines."""

    def __init__(self):
        self._nodes: Dict[State, Set[State]] = {}  # state -> children
        self._transitions: Dict[State, Set[Transition]] = {}
        self._parent_map: Dict[State, Optional[State]] = {}
        self._history = AsyncStateHistory()
        self._structure_lock = asyncio.Lock()
        self._transition_lock = asyncio.Lock()

    async def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """Thread-safe addition of states to the graph."""
        async with self._structure_lock:
            if state in self._nodes:
                return

            self._nodes[state] = set()
            self._parent_map[state] = parent
            self._transitions[state] = set()

            if parent:
                if parent not in self._nodes:
                    await self.add_state(parent)
                self._nodes[parent].add(state)

    async def add_transition(self, transition: Transition) -> None:
        """Thread-safe addition of transitions to the graph."""
        async with self._structure_lock:
            if transition.source not in self._nodes:
                raise ValueError(f"Source state {transition.source.name} not in graph")
            if transition.target not in self._nodes:
                raise ValueError(f"Target state {transition.target.name} not in graph")

            self._transitions[transition.source].add(transition)

    async def get_valid_transitions(self, state: State, event: Event) -> List[Transition]:
        """Thread-safe retrieval and evaluation of valid transitions."""
        async with self._transition_lock:
            if state not in self._transitions:
                return []
            return sorted(
                [t for t in self._transitions[state] if t.matches_event(event)],
                key=lambda t: t.get_priority(),
                reverse=True
            )

    async def record_history(self, composite_state: CompositeState, state: State) -> None:
        """Thread-safe history recording."""
        await self._history.record_state(composite_state, state)

    async def get_last_state(self, composite_state: CompositeState) -> Optional[State]:
        """Thread-safe history retrieval."""
        return await self._history.get_last_state(composite_state)

    async def clear_history(self) -> None:
        """Thread-safe history clearing."""
        await self._history.clear()

    def get_composite_ancestors(self, state: State) -> List[CompositeState]:
        """Get composite state ancestors (no locking needed - read-only operation)."""
        ancestors = []
        current = self._parent_map.get(state)
        while current:
            if isinstance(current, CompositeState):
                ancestors.append(current)
            current = self._parent_map.get(current)
        return ancestors

    async def validate(self) -> List[str]:
        """Thread-safe graph validation."""
        async with self._structure_lock:
            errors = []
            # Validate all states have proper parent relationships
            for state, parent in self._parent_map.items():
                if parent and parent not in self._nodes:
                    errors.append(f"Parent state {parent.name} of {state.name} not in graph")
                if isinstance(state, CompositeState) and not state._initial_state:
                    errors.append(f"Composite state {state.name} has no initial state")

            # Validate all transitions reference valid states
            for source, transitions in self._transitions.items():
                for transition in transitions:
                    if transition.target not in self._nodes:
                        errors.append(
                            f"Transition target {transition.target.name} from {source.name} not in graph"
                        )
            return errors


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


class AsyncStateMachine:
    """
    Asynchronous version of the state machine that supports async event processing
    with proper thread safety.
    """

    def __init__(self, initial_state: State, validator: Optional[Validator] = None, hooks: Optional[List] = None):
        self._graph = AsyncStateGraph()
        self._validator = validator or Validator()
        self._hooks = hooks or []
        self._async_lock = asyncio.Lock()
        self._started = False
        self._initial_state = initial_state
        self._current_state: Optional[State] = None

    async def add_state(self, state: State, parent: Optional[State] = None) -> None:
        """Add a state to the graph with proper locking."""
        await self._graph.add_state(state, parent)
        if isinstance(parent, CompositeState) and not parent._initial_state:
            parent._initial_state = state

    async def add_transition(self, transition: Transition) -> None:
        """Add a transition to the graph with proper locking."""
        await self._graph.add_transition(transition)

    @property
    def current_state(self) -> Optional[State]:
        """Get the current state (no locking needed - atomic read)."""
        return self._current_state

    async def _set_current_state(self, state: Optional[State], notify: bool = True) -> None:
        """Thread-safe current state update with optional notifications."""
        async with self._async_lock:
            if notify and self._current_state:
                await self._notify_exit_async(self._current_state)

            self._current_state = state

            if notify and state:
                await self._notify_enter_async(state)

    async def start(self) -> None:
        """Start the state machine with proper locking."""
        async with self._async_lock:
            if self._started:
                return

            # Validate the graph structure
            errors = await self._graph.validate()
            if errors:
                raise ValidationError("\n".join(errors))

            # Validator may be async or sync
            if asyncio.iscoroutinefunction(self._validator.validate_state_machine):
                await self._validator.validate_state_machine(self)
            else:
                self._validator.validate_state_machine(self)

            # Resolve initial state
            resolved_state = await self._resolve_active_state(self._initial_state)
            await self._set_current_state(resolved_state, notify=True)

            self._started = True

    async def stop(self) -> None:
        """Stop the state machine with proper locking."""
        async with self._async_lock:
            if not self._started:
                return

            if self._current_state:
                # Record history for composite ancestors
                ancestors = self._graph.get_composite_ancestors(self._current_state)
                if ancestors:
                    await self._graph.record_history(ancestors[0], self._current_state)

                await self._set_current_state(None, notify=True)

            self._started = False

    async def process_event(self, event: Event) -> bool:
        """Process an event with proper locking and transition handling."""
        if not self._started or not self._current_state:
            return False

        async with self._async_lock:
            try:
                # Get valid transitions with proper locking
                transitions = await self._graph.get_valid_transitions(self._current_state, event)
                if not transitions:
                    return False

                # Evaluate guards
                valid_transitions = []
                for transition in transitions:
                    if await self._evaluate_guards(transition, event):
                        valid_transitions.append(transition)
                        break  # Take first valid transition

                if not valid_transitions:
                    return False

                # Execute the transition
                transition = valid_transitions[0]
                result = await self._execute_transition_async(transition, event)

                # Handle composite state entry
                if isinstance(transition.target, CompositeState):
                    initial_state = transition.target._initial_state
                    if initial_state:
                        initial_transition = Transition(
                            source=transition.target,
                            target=initial_state,
                            guards=[lambda e: True]
                        )
                        await self._execute_transition_async(initial_transition, event)

                return result if result is not None else True

            except Exception as error:
                await self._notify_error_async(error)
                raise

    async def _evaluate_guards(self, transition: Transition, event: Event) -> bool:
        """Evaluate transition guards with proper async handling."""
        for guard in transition.guards:
            if asyncio.iscoroutinefunction(guard):
                if not await guard(event):
                    return False
            elif not guard(event):
                return False
        return True

    async def _execute_transition_async(self, transition: Transition, event: Event) -> Optional[bool]:
        """Execute a transition with proper locking and history management."""
        previous_state = self._current_state
        try:
            # Exit current state hierarchy
            if self._current_state:
                current = self._current_state
                while current and current != transition.source:
                    await self._notify_exit_async(current)
                    if isinstance(current, CompositeState):
                        await self._graph.record_history(current, self._current_state)
                    current = current.parent

            # Execute transition actions
            for action in transition.actions:
                if asyncio.iscoroutinefunction(action):
                    await action(event)
                else:
                    action(event)

            # Update current state
            await self._set_current_state(transition.target, notify=True)

            return True

        except Exception as e:
            # Restore previous state if transition failed
            await self._set_current_state(previous_state, notify=False)
            await self._notify_error_async(e)
            return False

    async def _resolve_active_state(self, state: State) -> State:
        """Resolve the active state using history if available."""
        if isinstance(state, CompositeState):
            hist_state = await self._graph.get_last_state(state)
            if hist_state:
                return await self._resolve_active_state(hist_state)
            if state._initial_state:
                return await self._resolve_active_state(state._initial_state)
        return state

    async def _notify_enter_async(self, state: State) -> None:
        """Notify enter with proper async handling."""
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
        """Notify exit with proper async handling."""
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
        """Notify error with proper async handling."""
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
        self._submachine_lock = asyncio.Lock()

    async def add_submachine(self, state: CompositeState, submachine: "AsyncStateMachine") -> None:
        """
        Add a submachine's states under a parent composite state with proper locking.
        """
        if not isinstance(state, CompositeState):
            raise ValueError(f"State {state.name} must be a composite state")

        async with self._submachine_lock:
            # First add the composite state if it's not already in the graph
            if state not in self._graph._nodes:
                await self._graph.add_state(state)

            # Integrate submachine states into the same graph
            for sub_state in submachine.current_state._graph._nodes:
                await self._graph.add_state(sub_state, parent=state)

            # Integrate transitions
            for source, transitions in submachine._graph._transitions.items():
                for transition in transitions:
                    await self._graph.add_transition(transition)

            # Set the composite state's initial state
            if submachine._initial_state:
                state._initial_state = submachine._initial_state
                # Add a transition from the composite state to its initial state
                await self._graph.add_transition(
                    Transition(source=state, target=submachine._initial_state, guards=[lambda e: True])
                )

            self._submachines[state] = submachine

    async def process_event(self, event: Event) -> bool:
        """
        Process events with proper handling of submachine hierarchy and async locking.
        """
        if not self._started or not self._current_state:
            return False

        async with self._async_lock:
            try:
                # First try transitions from the current state and its ancestors,
                # but stop if we hit a composite state that's a submachine root
                current = self._current_state
                current_transitions = []
                while current:
                    if current in self._submachines:
                        break
                    transitions = await self._graph.get_valid_transitions(current, event)
                    current_transitions.extend(transitions)
                    current = current.parent

                # Now check transitions from submachine boundaries
                boundary_transitions = []
                current = self._current_state
                while current:
                    if current.parent in self._submachines:
                        transitions = await self._graph.get_valid_transitions(current.parent, event)
                        boundary_transitions.extend(transitions)
                    current = current.parent

                # Evaluate transitions in order of precedence
                potential_transitions = current_transitions + boundary_transitions
                if not potential_transitions:
                    return False

                # Evaluate guards and take first valid transition
                valid_transition = None
                for transition in potential_transitions:
                    if await self._evaluate_guards(transition, event):
                        valid_transition = transition
                        break

                if not valid_transition:
                    return False

                # Execute the transition with proper history management
                result = await self._execute_hierarchical_transition(valid_transition, event)
                return result if result is not None else True

            except Exception as error:
                await self._notify_error_async(error)
                raise

    async def _execute_hierarchical_transition(self, transition: Transition, event: Event) -> Optional[bool]:
        """Execute a transition with proper handling of submachine hierarchy."""
        previous_state = self._current_state
        try:
            # Exit current state hierarchy up to transition source
            if self._current_state:
                current = self._current_state
                while current and current != transition.source:
                    await self._notify_exit_async(current)
                    if isinstance(current, CompositeState):
                        await self._graph.record_history(current, self._current_state)
                    current = current.parent

            # Execute transition actions
            for action in transition.actions:
                if asyncio.iscoroutinefunction(action):
                    await action(event)
                else:
                    action(event)

            # Update current state
            await self._set_current_state(transition.target, notify=True)

            # If target is a composite state, enter its initial state
            if isinstance(transition.target, CompositeState):
                submachine = self._submachines.get(transition.target)
                if submachine:
                    initial_state = submachine._initial_state
                    if initial_state:
                        initial_transition = Transition(
                            source=transition.target,
                            target=initial_state,
                            guards=[lambda e: True]
                        )
                        await self._execute_transition_async(initial_transition, event)
                else:
                    initial_state = transition.target._initial_state
                    if initial_state:
                        initial_transition = Transition(
                            source=transition.target,
                            target=initial_state,
                            guards=[lambda e: True]
                        )
                        await self._execute_transition_async(initial_transition, event)

            return True

        except Exception as e:
            # Restore previous state if transition failed
            await self._set_current_state(previous_state, notify=False)
            await self._notify_error_async(e)
            return False
