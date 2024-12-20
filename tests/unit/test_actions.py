# tests/unit/test_actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_basic_actions_execute():
    from hsm.core.actions import BasicActions

    action_called = False

    def action_fn(arg=None):
        nonlocal action_called
        action_called = True
        assert arg == "test"

    BasicActions.execute(action_fn, arg="test")
    assert action_called is True
