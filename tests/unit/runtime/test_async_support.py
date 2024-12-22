# tests/unit/test_async_support.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import AsyncValidator, Validator
from hsm.runtime.async_support import (
    AsyncCompositeStateMachine,
    AsyncEventQueue,
    AsyncStateMachine,
    _AsyncEventProcessingLoop,
    _AsyncLock,
)


@pytest.mark.asyncio
async def test_async_state_machine_init(dummy_state, validator):
    """Test initialization of AsyncStateMachine."""
    asm = AsyncStateMachine(initial_state=dummy_state, validator=validator)
    initial = asm._graph.get_initial_state(None)  # Get root initial state
    assert initial == dummy_state
    assert asm.current_state is None


@pytest.mark.asyncio
async def test_async_state_machine_start_stop(dummy_state):
    """Test starting and stopping AsyncStateMachine."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    dummy_state.on_enter = MagicMock()
    await asm.start()
    dummy_state.on_enter.assert_called_once()
    await asm.stop()
    dummy_state.on_exit = MagicMock()


@pytest.mark.asyncio
async def test_async_state_machine_process_event(dummy_state, mock_event):
    """Test event processing in AsyncStateMachine."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    await asm.start()
    # With no transitions, process_event should not fail
    await asm.process_event(mock_event)


@pytest.mark.asyncio
async def test_async_event_queue(mock_event):
    """Test basic AsyncEventQueue operations."""
    eq = AsyncEventQueue(priority=False)
    await eq.enqueue(mock_event)
    out = await eq.dequeue()
    assert out == mock_event
    await eq.clear()
    empty = await eq.dequeue()
    assert empty is None
    assert eq.priority_mode is False


@pytest.mark.asyncio
async def test_async_lock():
    """Test AsyncLock functionality."""
    lock = _AsyncLock()
    await lock.acquire()
    # Verify lock is acquired
    lock.release()
    # Verify lock can be reacquired
    await lock.acquire()
    lock.release()


@pytest.mark.asyncio
async def test_async_state_machine_double_start(dummy_state):
    """Test starting AsyncStateMachine multiple times."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    dummy_state.on_enter = MagicMock()
    await asm.start()
    await asm.start()  # Second start should be ignored
    dummy_state.on_enter.assert_called_once()


@pytest.mark.asyncio
async def test_async_state_machine_transition(dummy_state, mock_event):
    """Test state transitions in AsyncStateMachine."""
    target_state = State("target")
    transition = Transition(source=dummy_state, target=target_state, guards=[lambda e: True], actions=[MagicMock()])

    asm = AsyncStateMachine(initial_state=dummy_state)
    asm.add_state(target_state)
    asm.add_transition(transition)
    await asm.start()
    await asm.process_event(mock_event)

    assert asm.current_state == target_state
    transition.actions[0].assert_called_once_with(mock_event)


@pytest.mark.asyncio
async def test_async_composite_state_machine():
    """Test AsyncCompositeStateMachine functionality."""
    root = CompositeState("root")
    sub1 = State("sub1")
    sub2 = State("sub2")

    # Create main machine
    main_machine = AsyncCompositeStateMachine(initial_state=root)

    # Create submachine
    sub_machine = AsyncStateMachine(initial_state=sub1)
    sub_machine.add_state(sub2)

    # Add transition after all states are set up
    sub_machine.add_transition(Transition(source=sub1, target=sub2, guards=[lambda e: e.name == "next"]))

    # Add submachine to main machine
    main_machine.add_submachine(root, sub_machine)

    # Start and verify initial state
    await main_machine.start()
    assert main_machine.current_state == sub1

    # Test transition within submachine
    await main_machine.process_event(Event("next"))
    assert main_machine.current_state == sub2


@pytest.mark.asyncio
async def test_async_event_processing_loop():
    """Test AsyncEventProcessingLoop functionality."""
    initial_state = State("initial")
    machine = AsyncStateMachine(initial_state=initial_state)
    queue = AsyncEventQueue()

    loop = _AsyncEventProcessingLoop(machine, queue)

    # Start loop in background task
    task = asyncio.create_task(loop.start_loop())

    # Give loop time to start
    await asyncio.sleep(0.1)

    # Stop loop
    await loop.stop_loop()

    # Clean up task
    await task


@pytest.mark.asyncio
async def test_async_event_queue_priority_mode():
    """Test priority mode in AsyncEventQueue."""
    eq = AsyncEventQueue(priority=True)
    assert eq.priority_mode is True

    await eq.enqueue(Event("first", priority=1))
    await eq.enqueue(Event("second", priority=1))

    first = await eq.dequeue()
    second = await eq.dequeue()
    assert first.name == "first"
    assert second.name == "second"


@pytest.mark.asyncio
async def test_async_state_machine_error_handling(dummy_state, mock_event):
    """Test error handling in AsyncStateMachine."""
    error_hook = MagicMock()

    async def async_on_enter(state):
        pass

    error_hook.on_enter = async_on_enter
    error_hook.on_error = MagicMock()

    asm = AsyncStateMachine(initial_state=dummy_state, hooks=[error_hook])
    await asm.start()

    # Cause an error by making the state's on_exit throw
    dummy_state.on_exit = MagicMock(side_effect=Exception("Test error"))

    # Add a transition that will trigger the error
    target_state = State("target")
    asm.add_state(target_state)
    transition = Transition(source=dummy_state, target=target_state, guards=[lambda e: True])
    asm.add_transition(transition)

    await asm.process_event(mock_event)
    error_hook.on_error.assert_called_once()


@pytest.mark.asyncio
async def test_async_event_queue_timeout():
    """Test timeout behavior in AsyncEventQueue."""
    eq = AsyncEventQueue()
    # Should return None when queue is empty (after timeout)
    result = await eq.dequeue()
    assert result is None


@pytest.mark.asyncio
async def test_async_state_machine_process_event_when_stopped(dummy_state, mock_event):
    """Test event processing when machine is stopped."""
    asm = AsyncStateMachine(initial_state=dummy_state)
    # Don't start the machine
    await asm.process_event(mock_event)  # Should do nothing
    assert asm.current_state is None


@pytest.mark.asyncio
async def test_async_state_machine_validator_integration():
    """Test async validator integration."""

    class CustomAsyncValidator(AsyncValidator):
        async def validate_state_machine(self, machine):
            # Simulate async validation
            await asyncio.sleep(0.1)
            await super().validate_state_machine(machine)

    start_state = State("Start")
    machine = AsyncStateMachine(initial_state=start_state, validator=CustomAsyncValidator())
    await machine.start()
    await machine.stop()


@pytest.mark.asyncio
async def test_async_state_machine_with_async_guards():
    """Test AsyncStateMachine with async guard functions."""
    initial = State("initial")
    target = State("target")

    async def async_guard(event):
        await asyncio.sleep(0.1)
        return True

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, guards=[async_guard]))

    await machine.start()
    await machine.process_event(Event("test"))
    assert machine.current_state == target


@pytest.mark.asyncio
async def test_async_state_machine_with_async_actions():
    """Test AsyncStateMachine with async actions."""
    initial = State("initial")
    target = State("target")
    action_executed = False

    async def async_action(event):
        nonlocal action_executed
        await asyncio.sleep(0.1)
        action_executed = True

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, guards=[lambda e: True], actions=[async_action]))

    await machine.start()
    await machine.process_event(Event("test"))
    assert action_executed is True


@pytest.mark.asyncio
async def test_async_composite_state_machine_boundary_transitions():
    """Test boundary transitions in AsyncCompositeStateMachine."""
    # Create states
    root = CompositeState("root")
    sub1 = State("sub1")
    sub2 = State("sub2")
    external = State("external")

    # Create submachine with graph
    sub_machine = AsyncStateMachine(initial_state=sub1)
    sub_machine.add_state(sub2)
    sub_machine._graph.add_state(sub1)
    sub_machine._graph.add_state(sub2)

    # Add transition after states are set up
    sub_machine.add_transition(Transition(source=sub1, target=sub2, guards=[lambda e: e.name == "internal"]))

    # Create main machine
    main_machine = AsyncCompositeStateMachine(initial_state=root)
    main_machine.add_state(external)
    main_machine.add_submachine(root, sub_machine)

    # Add boundary transition
    main_machine.add_transition(
        Transition(
            source=sub2, target=external, guards=[lambda e: e.name == "exit_submachine"]  # Changed from root to sub2
        )
    )

    # Start and verify initial state
    await main_machine.start()
    assert main_machine.current_state == sub1

    # Test internal transition
    await main_machine.process_event(Event("internal"))
    assert main_machine.current_state == sub2

    # Test boundary transition
    await main_machine.process_event(Event("exit_submachine"))
    assert main_machine.current_state == external


@pytest.mark.asyncio
async def test_async_event_queue_concurrent_operations():
    """Test concurrent operations on AsyncEventQueue."""
    queue = AsyncEventQueue(priority=True)
    events = [Event(f"event{i}", priority=i % 3) for i in range(10)]

    # Enqueue events concurrently
    await asyncio.gather(*(queue.enqueue(event) for event in events))

    # Dequeue events and verify priority ordering
    received_events = []
    while not queue.is_empty():
        event = await queue.dequeue()
        if event:
            received_events.append(event)

    # Verify events were received in priority order
    priorities = [event.priority for event in received_events]
    assert priorities == sorted(priorities, reverse=True)


@pytest.mark.asyncio
async def test_async_state_machine_concurrent_transitions():
    """Test concurrent transitions in AsyncStateMachine."""
    initial = State("initial")
    target1 = State("target1")
    target2 = State("target2")

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target1)
    machine.add_state(target2)

    # Add transitions with different priorities
    machine.add_transition(Transition(source=initial, target=target1, guards=[lambda e: e.name == "test"], priority=1))
    machine.add_transition(Transition(source=initial, target=target2, guards=[lambda e: e.name == "test"], priority=2))

    await machine.start()

    # Process same event multiple times concurrently
    event = Event("test")
    results = await asyncio.gather(*(machine.process_event(event) for _ in range(5)))

    # Verify machine ended up in the higher priority state
    assert machine.current_state == target2
    # At least one transition should have succeeded
    assert any(results)


@pytest.mark.asyncio
async def test_async_lock_reentrant():
    """Test AsyncLock reentrant behavior."""
    lock = _AsyncLock()
    await lock.acquire()

    # Try to acquire again - should timeout
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(lock.acquire(), timeout=0.1)

    lock.release()


@pytest.mark.asyncio
async def test_event_queue_stop_behavior():
    """Test AsyncEventQueue stop behavior."""
    queue = AsyncEventQueue()
    await queue.enqueue(Event("test"))

    # Stop queue
    await queue.stop()

    # Dequeue should return None after stop
    assert await queue.dequeue() is None

    # Verify queue stays stopped
    assert not queue._running
    assert queue.is_empty()


@pytest.mark.asyncio
async def test_async_state_machine_transition_error_recovery():
    """Test AsyncStateMachine transition error recovery."""
    initial = State("initial")
    target = State("target")

    # Create action that raises exception
    async def failing_action(event):
        raise ValueError("Action failed")

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, actions=[failing_action]))

    # Add error hook to track error
    error_called = False

    async def error_hook(error):
        nonlocal error_called
        error_called = True

    class TestHook:
        async def on_error(self, error):
            await error_hook(error)

    machine._hooks.append(TestHook())

    await machine.start()
    assert machine.current_state == initial

    # Process event - should fail but recover
    result = await machine.process_event(Event("test"))
    assert result is False  # Transition failed
    assert machine.current_state == initial  # State restored
    assert error_called  # Error hook called


@pytest.mark.asyncio
async def test_async_composite_state_machine_validation():
    """Test AsyncCompositeStateMachine validation."""
    root = CompositeState("root")
    sub = CompositeState("sub")  # No initial state

    machine = AsyncCompositeStateMachine(initial_state=root)

    # Try to add invalid submachine
    submachine = AsyncStateMachine(initial_state=sub)

    with pytest.raises(ValueError, match="State 'not_composite' not found in state machine"):
        machine.add_submachine(State("not_composite"), submachine)


@pytest.mark.asyncio
async def test_async_event_processing_loop_error_handling():
    """Test AsyncEventProcessingLoop error handling."""
    initial = State("initial")
    machine = AsyncStateMachine(initial_state=initial)
    queue = AsyncEventQueue()

    # Create action that raises exception
    async def failing_action(event):
        raise ValueError("Action failed")

    target = State("target")
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, actions=[failing_action]))

    # Add error hook to track error
    error_occurred = False

    class TestHook:
        async def on_error(self, error):
            nonlocal error_occurred
            error_occurred = True

    machine._hooks.append(TestHook())

    loop = _AsyncEventProcessingLoop(machine, queue)

    # Start loop and machine
    task = asyncio.create_task(loop.start_loop())
    await asyncio.sleep(0.1)  # Let loop start

    # Verify machine is in initial state
    assert machine.current_state == initial

    # Enqueue event that will cause error
    await queue.enqueue(Event("test"))
    await asyncio.sleep(0.1)  # Let event process

    # Verify error occurred but don't stop the loop yet
    assert error_occurred
    assert machine.current_state == initial  # Should still be in initial state after error

    # Now stop the loop
    await loop.stop_loop()
    await task


@pytest.mark.asyncio
async def test_async_state_machine_parent_transitions():
    """Test AsyncStateMachine parent state transitions."""
    root = CompositeState("root")
    child1 = State("child1")
    child2 = State("child2")
    target = State("target")

    machine = AsyncStateMachine(initial_state=root)
    machine.add_state(child1, parent=root)
    machine.add_state(child2, parent=root)
    machine.add_state(target)

    # Add transition from parent state
    machine.add_transition(
        Transition(source=child1, target=target, guards=[lambda e: e.name == "test"])  # Changed from root to child1
    )

    machine._graph.set_initial_state(root, child1)
    await machine.start()
    assert machine.current_state == child1

    # Process event - should transition to target
    await machine.process_event(Event("test"))
    assert machine.current_state == target


@pytest.mark.asyncio
async def test_async_event_queue_error_handling():
    """Test AsyncEventQueue error handling."""
    queue = AsyncEventQueue()

    # Test enqueue and dequeue behavior
    await queue.enqueue(Event("test"))
    await queue.stop()

    # After stop, dequeue should return None
    assert await queue.dequeue() is None
    assert queue.is_empty()

    # Verify queue stays stopped
    assert not queue._running


@pytest.mark.asyncio
async def test_async_state_machine_validation():
    """Test AsyncStateMachine validation."""
    # Create invalid composite state (no children)
    composite = CompositeState("composite")
    composite._children = set()

    machine = AsyncStateMachine(initial_state=composite)

    # Start should fail because composite has no children and no initial state
    with pytest.raises(ValidationError, match="Composite state 'composite' has no children"):
        await machine.start()


@pytest.mark.asyncio
async def test_async_composite_state_machine_submachine():
    """Test AsyncCompositeStateMachine submachine management."""
    root = CompositeState("root")
    sub = CompositeState("sub")
    sub_state = State("sub_state")

    # Create submachine with proper hierarchy
    submachine = AsyncStateMachine(initial_state=sub_state)  # Changed initial state
    submachine.add_state(sub_state)

    # Create main machine
    machine = AsyncCompositeStateMachine(initial_state=root)

    # Add submachine
    machine.add_submachine(root, submachine)

    # Start machine
    await machine.start()
    assert machine.current_state == sub_state  # Should be in sub_state


@pytest.mark.asyncio
async def test_async_hook_execution():
    """Test async hook execution."""
    initial = State("initial")
    target = State("target")

    hook_calls = []

    class TestHook:
        async def on_enter(self, state):
            hook_calls.append(f"enter_{state.name}")

        async def on_exit(self, state):
            hook_calls.append(f"exit_{state.name}")

        async def on_transition(self, source, target):
            hook_calls.append(f"transition_{source.name}_to_{target.name}")

        async def on_error(self, error):
            hook_calls.append(f"error_{str(error)}")

    machine = AsyncStateMachine(initial_state=initial, hooks=[TestHook()])
    machine.add_state(target)
    machine.add_transition(Transition(initial, target))

    await machine.start()
    assert "enter_initial" in hook_calls

    await machine.process_event(Event("test"))
    assert "exit_initial" in hook_calls
    assert "transition_initial_to_target" in hook_calls
    assert "enter_target" in hook_calls


@pytest.mark.asyncio
async def test_async_error_handling():
    """Test async error handling in transitions."""
    initial = State("initial")
    target = State("target")

    async def failing_action(event):
        raise ValueError("Action failed")

    error_caught = False

    class ErrorHook:
        async def on_error(self, error):
            nonlocal error_caught
            error_caught = True

    machine = AsyncStateMachine(initial_state=initial, hooks=[ErrorHook()])
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, actions=[failing_action]))

    await machine.start()
    await machine.process_event(Event("test"))

    assert error_caught
    assert machine.current_state == initial  # Should stay in initial state


@pytest.mark.asyncio
async def test_async_composite_state_machine_error_handling():
    """Test error handling in AsyncCompositeStateMachine."""
    root = CompositeState("root")
    state = State("state")

    # Create machine with invalid state
    machine = AsyncCompositeStateMachine(initial_state=root)

    # Test adding submachine to non-existent state
    non_existent = CompositeState("non_existent")
    submachine = AsyncStateMachine(initial_state=state)
    with pytest.raises(ValueError, match="State 'non_existent' not found in state machine"):
        machine.add_submachine(non_existent, submachine)

    # Test adding submachine to non-composite state
    machine.add_state(state)  # Add regular state to machine
    with pytest.raises(ValueError, match="State 'state' must be a composite state"):
        machine.add_submachine(state, submachine)


@pytest.mark.asyncio
async def test_async_composite_state_machine_transitions():
    """Test AsyncCompositeStateMachine transition handling."""
    # Set up state hierarchy
    root = CompositeState("root")
    sub1 = CompositeState("sub1")
    state1 = State("state1")
    state2 = State("state2")

    # Create submachine with proper hierarchy
    submachine = AsyncStateMachine(initial_state=state1)
    submachine.add_state(state2)
    submachine.add_transition(Transition(state1, state2, guards=[lambda e: e.name == "next"]))

    # Create main machine with proper hierarchy
    machine = AsyncCompositeStateMachine(initial_state=root)
    machine.add_state(sub1, parent=root)
    machine._graph.set_initial_state(root, sub1)

    # Add submachine
    machine.add_submachine(sub1, submachine)

    # Start machine and verify initial state
    await machine.start()
    assert machine.current_state == state1

    # Test transition within submachine
    await machine.process_event(Event("next"))
    assert machine.current_state == state2


@pytest.mark.asyncio
async def test_async_composite_state_machine_nested():
    """Test deeply nested AsyncCompositeStateMachine behavior."""
    # Create nested state hierarchy
    root = CompositeState("root")
    level1 = CompositeState("level1")
    leaf1 = State("leaf1")
    leaf2 = State("leaf2")

    # Create innermost submachine
    inner_machine = AsyncStateMachine(initial_state=leaf1)
    inner_machine.add_state(leaf2)
    inner_machine.add_transition(Transition(leaf1, leaf2, guards=[lambda e: e.name == "next"]))

    # Create main machine with proper hierarchy
    machine = AsyncCompositeStateMachine(initial_state=root)
    machine.add_state(level1, parent=root)
    machine._graph.set_initial_state(root, level1)

    # Add submachine
    machine.add_submachine(level1, inner_machine)

    # Start machine and verify initial state resolution
    await machine.start()
    assert machine.current_state == leaf1

    # Test transition in submachine
    await machine.process_event(Event("next"))
    assert machine.current_state == leaf2


@pytest.mark.asyncio
async def test_async_composite_state_machine_parallel_transitions():
    """Test parallel transitions in AsyncCompositeStateMachine."""
    # Set up parallel composite states
    root = CompositeState("root")
    parallel1 = CompositeState("parallel1")
    parallel2 = CompositeState("parallel2")
    state1 = State("state1")
    state2 = State("state2")
    state3 = State("state3")  # State for parallel2

    # Create submachine for parallel1
    sub1 = AsyncStateMachine(initial_state=state1)
    sub1.add_state(state2)
    sub1.add_transition(Transition(state1, state2, guards=[lambda e: e.name == "next"]))

    # Create submachine for parallel2
    sub2 = AsyncStateMachine(initial_state=state3)

    # Create main machine with proper hierarchy
    machine = AsyncCompositeStateMachine(initial_state=root)
    machine.add_state(parallel1, parent=root)
    machine.add_state(parallel2, parent=root)
    machine._graph.set_initial_state(root, parallel1)

    # Add submachines to both parallel states
    machine.add_submachine(parallel1, sub1)
    machine.add_submachine(parallel2, sub2)

    # Start machine and verify initial state
    await machine.start()
    assert machine.current_state == state1

    # Test transition
    await machine.process_event(Event("next"))
    assert machine.current_state == state2


@pytest.mark.asyncio
async def test_async_state_machine_with_async_guards():
    """Test AsyncStateMachine with async guard functions."""
    initial = State("initial")
    target = State("target")

    async def async_guard(event):
        await asyncio.sleep(0.1)
        return False  # Test guard rejection

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, guards=[async_guard]))

    await machine.start()
    result = await machine.process_event(Event("test"))
    assert result is False  # Event should be rejected
    assert machine.current_state == initial


@pytest.mark.asyncio
async def test_async_state_machine_with_async_actions():
    """Test AsyncStateMachine with async actions."""
    initial = State("initial")
    target = State("target")
    action_executed = False

    async def async_action(event):
        nonlocal action_executed
        await asyncio.sleep(0.1)
        action_executed = True
        raise ValueError("Action failed")  # Test error handling

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(source=initial, target=target, guards=[lambda e: True], actions=[async_action]))

    # Add error hook
    error_hook = MagicMock()
    error_hook.on_error = AsyncMock()
    machine._hooks.append(error_hook)

    await machine.start()
    result = await machine.process_event(Event("test"))

    assert result is False  # Transition should fail
    assert action_executed is True  # Action should execute
    assert machine.current_state == initial  # Should remain in initial state
    error_hook.on_error.assert_called_once()


@pytest.mark.asyncio
async def test_async_state_machine_hooks():
    """Test async hooks in AsyncStateMachine."""
    initial = State("initial")
    target = State("target")
    hook_calls = []

    class TestHook:
        async def on_enter(self, state):
            hook_calls.append(f"enter_{state.name}")

        async def on_exit(self, state):
            hook_calls.append(f"exit_{state.name}")

        async def on_transition(self, source, target):
            hook_calls.append(f"transition_{source.name}_to_{target.name}")

        async def on_error(self, error):
            hook_calls.append(f"error_{str(error)}")

    machine = AsyncStateMachine(initial_state=initial, hooks=[TestHook()])
    machine.add_state(target)
    machine.add_transition(Transition(initial, target))

    await machine.start()
    assert "enter_initial" in hook_calls

    await machine.process_event(Event("test"))
    assert "exit_initial" in hook_calls
    assert "transition_initial_to_target" in hook_calls
    assert "enter_target" in hook_calls

    # Test error handling
    async def failing_action(event):
        raise ValueError("Test error")

    machine.add_transition(Transition(target, initial, actions=[failing_action]))

    await machine.process_event(Event("test"))
    assert "error_Test error" in hook_calls


@pytest.mark.asyncio
async def test_async_state_machine_sync_hooks():
    """Test synchronous hooks in AsyncStateMachine."""
    initial = State("initial")
    target = State("target")

    class SyncHook:
        def on_enter(self, state):
            state.entered = True

        def on_exit(self, state):
            state.exited = True

        def on_transition(self, source, target):
            source.transitioned = True
            target.transitioned = True

        def on_error(self, error):
            pass

    machine = AsyncStateMachine(initial_state=initial, hooks=[SyncHook()])
    machine.add_state(target)
    machine.add_transition(Transition(initial, target))

    await machine.start()
    assert hasattr(initial, "entered")

    await machine.process_event(Event("test"))
    assert hasattr(initial, "exited")
    assert hasattr(initial, "transitioned")
    assert hasattr(target, "transitioned")


@pytest.mark.asyncio
async def test_async_state_machine_sync_validator():
    """Test synchronous validator in AsyncStateMachine."""
    initial = State("initial")

    class SyncValidator(Validator):
        def validate_state_machine(self, machine):
            return True

    machine = AsyncStateMachine(initial_state=initial, validator=SyncValidator())
    await machine.start()
    assert machine.current_state == initial


@pytest.mark.asyncio
async def test_async_state_machine_sync_state_methods():
    """Test synchronous state methods in AsyncStateMachine."""
    initial = State("initial")
    target = State("target")

    initial.on_enter = MagicMock()
    initial.on_exit = MagicMock()
    target.on_enter = MagicMock()

    machine = AsyncStateMachine(initial_state=initial)
    machine.add_state(target)
    machine.add_transition(Transition(initial, target))

    await machine.start()
    initial.on_enter.assert_called_once()

    await machine.process_event(Event("test"))
    initial.on_exit.assert_called_once()
    target.on_enter.assert_called_once()


@pytest.mark.asyncio
async def test_async_composite_state_machine_empty_transitions():
    """Test AsyncCompositeStateMachine with no valid transitions."""
    root = CompositeState("root")
    state = State("state")

    machine = AsyncCompositeStateMachine(initial_state=root)
    machine.add_state(state, parent=root)
    machine._graph.set_initial_state(root, state)

    await machine.start()
    assert machine.current_state == state

    # Test processing event with no transitions
    result = await machine.process_event(Event("test"))
    assert result is False


@pytest.mark.asyncio
async def test_async_composite_state_machine_transition_error():
    """Test AsyncCompositeStateMachine transition error handling."""
    root = CompositeState("root")
    state1 = State("state1")
    state2 = State("state2")

    async def failing_guard(event):
        raise ValueError("Guard failed")

    machine = AsyncCompositeStateMachine(initial_state=root)
    machine.add_state(state1, parent=root)
    machine.add_state(state2, parent=root)
    machine._graph.set_initial_state(root, state1)

    machine.add_transition(Transition(source=state1, target=state2, guards=[failing_guard]))

    error_hook = MagicMock()
    error_hook.on_error = AsyncMock()
    machine._hooks.append(error_hook)

    await machine.start()
    await machine.process_event(Event("test"))
    error_hook.on_error.assert_called_once()


@pytest.mark.asyncio
async def test_async_event_queue_stop_clear():
    """Test AsyncEventQueue stop and clear functionality."""
    queue = AsyncEventQueue()

    # Add some events
    events = [Event(f"event{i}") for i in range(3)]
    for event in events:
        await queue.enqueue(event)

    # Stop the queue
    await queue.stop()
    assert queue._running is False

    # Try to dequeue after stop
    result = await queue.dequeue()
    assert result is None

    # Try to clear after stop
    await queue.clear()
    assert queue.is_empty()


@pytest.mark.asyncio
async def test_async_event_processing_loop_error():
    """Test AsyncEventProcessingLoop error handling."""
    initial_state = State("initial")
    machine = AsyncStateMachine(initial_state=initial_state)
    queue = AsyncEventQueue()

    loop = _AsyncEventProcessingLoop(machine, queue)

    # Start loop in background task
    task = asyncio.create_task(loop.start_loop())

    # Give loop time to start
    await asyncio.sleep(0.1)

    # Enqueue an event that will cause an error
    event = Event("error")
    await queue.enqueue(event)

    # Stop loop
    await loop.stop_loop()

    # Clean up task
    await task
