# tests/unit/test_transitions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest


def test_transition_init(dummy_state, dummy_guard, dummy_action):
    from hsm.core.transitions import Transition

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
    from hsm.core.transitions import Transition

    def true_guard(e):
        return True

    def false_guard(e):
        return False

    t = Transition(dummy_state, dummy_state, guards=[true_guard, false_guard])
    result = await t.evaluate_guards(dummy_event)
    assert result is False


@pytest.mark.asyncio
async def test_transition_execute_actions(dummy_state, dummy_event):
    from hsm.core.transitions import Transition

    action_called = False

    def action_fn(event):
        nonlocal action_called
        action_called = True

    t = Transition(dummy_state, dummy_state, actions=[action_fn])
    await t.execute_actions(dummy_event)
    assert action_called is True
