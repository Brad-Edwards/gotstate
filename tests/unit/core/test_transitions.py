# tests/unit/test_transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest


def test_transition_init(dummy_state, dummy_guard, dummy_action):
    from gotstate.core.transitions import Transition

    t = Transition(
        source=dummy_state,
        target=dummy_state,
        guards=[dummy_guard],
        actions=[dummy_action],
        priority=5,
    )
    assert t.get_priority() == 5
    assert t.source == dummy_state
    assert t.target == dummy_state


@pytest.mark.asyncio
async def test_transition_evaluate_guards(dummy_state, dummy_event):
    from gotstate.core.transitions import Transition

    def true_guard(e):
        return True

    def false_guard(e):
        return False

    t = Transition(dummy_state, dummy_state, guards=[true_guard, false_guard])
    result = await t.evaluate_guards(dummy_event)
    assert result is False


@pytest.mark.asyncio
async def test_transition_execute_actions(dummy_state, dummy_event):
    from gotstate.core.transitions import Transition

    action_called = False

    def action_fn(event):
        nonlocal action_called
        action_called = True

    t = Transition(dummy_state, dummy_state, actions=[action_fn])
    await t.execute_actions(dummy_event)
    assert action_called is True


@pytest.mark.asyncio
async def test_async_guard_evaluation(dummy_state, dummy_event):
    """Test evaluation of async guard functions."""
    from gotstate.core.transitions import Transition

    async def async_true_guard(e):
        return True

    async def async_false_guard(e):
        return False

    # Test async true guard
    t1 = Transition(dummy_state, dummy_state, guards=[async_true_guard])
    assert await t1.evaluate_guards(dummy_event) is True

    # Test async false guard
    t2 = Transition(dummy_state, dummy_state, guards=[async_false_guard])
    assert await t2.evaluate_guards(dummy_event) is False

    # Test mixed sync/async guards
    def sync_true_guard(e):
        return True

    t3 = Transition(dummy_state, dummy_state, guards=[async_true_guard, sync_true_guard])
    assert await t3.evaluate_guards(dummy_event) is True


@pytest.mark.asyncio
async def test_async_action_execution(dummy_state, dummy_event):
    """Test execution of async actions."""
    from gotstate.core.errors import TransitionError
    from gotstate.core.transitions import Transition

    action_called = False

    async def async_action(e):
        nonlocal action_called
        action_called = True

    # Test async action
    t = Transition(dummy_state, dummy_state, actions=[async_action])
    await t.execute_actions(dummy_event)
    assert action_called is True

    # Test mixed sync/async actions
    action_called = False
    sync_called = False

    def sync_action(e):
        nonlocal sync_called
        sync_called = True

    t = Transition(dummy_state, dummy_state, actions=[async_action, sync_action])
    await t.execute_actions(dummy_event)
    assert action_called is True
    assert sync_called is True


@pytest.mark.asyncio
async def test_action_error_handling(dummy_state, dummy_event):
    """Test error handling during action execution."""
    from gotstate.core.errors import TransitionError
    from gotstate.core.transitions import Transition

    def failing_action(e):
        raise ValueError("Action failed")

    async def async_failing_action(e):
        raise ValueError("Async action failed")

    # Test sync action failure
    t1 = Transition(dummy_state, dummy_state, actions=[failing_action])
    with pytest.raises(TransitionError, match="Action execution failed: Action failed"):
        await t1.execute_actions(dummy_event)

    # Test async action failure
    t2 = Transition(dummy_state, dummy_state, actions=[async_failing_action])
    with pytest.raises(TransitionError, match="Action execution failed: Async action failed"):
        await t2.execute_actions(dummy_event)


def test_transition_priority_sorting():
    """Test sorting transitions by priority."""
    from gotstate.core.states import State
    from gotstate.core.transitions import Transition, _TransitionPrioritySorter

    state = State("test")
    t1 = Transition(state, state, priority=1)
    t2 = Transition(state, state, priority=3)
    t3 = Transition(state, state, priority=2)

    sorter = _TransitionPrioritySorter()
    sorted_transitions = sorter.sort([t1, t2, t3])

    assert sorted_transitions == [t2, t3, t1]  # Sorted by priority (highest first)


def test_transition_properties():
    """Test transition property access."""
    from gotstate.core.states import State
    from gotstate.core.transitions import Transition

    source = State("source")
    target = State("target")
    guard = lambda e: True
    action = lambda e: None

    t = Transition(source, target, guards=[guard], actions=[action], priority=5)

    assert t.source == source
    assert t.target == target
    assert t.guards == [guard]
    assert t.actions == [action]
    assert t.get_priority() == 5
