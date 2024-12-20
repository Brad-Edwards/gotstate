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


def test_action_adapter_init():
    from hsm.core.actions import _ActionAdapter

    action_called = False

    def test_action(event):
        nonlocal action_called
        action_called = True

    adapter = _ActionAdapter(test_action)
    assert adapter._action_fn == test_action


def test_action_adapter_run():
    from hsm.core.actions import _ActionAdapter
    from hsm.core.events import Event

    action_event = None

    def test_action(event):
        nonlocal action_event
        action_event = event

    adapter = _ActionAdapter(test_action)
    test_event = Event("TestEvent")
    adapter.run(test_event)
    assert action_event == test_event


def test_basic_actions_execute_no_args():
    from hsm.core.actions import BasicActions

    action_called = False

    def action_fn():
        nonlocal action_called
        action_called = True

    BasicActions.execute(action_fn)
    assert action_called is True


def test_basic_actions_execute_multiple_args():
    from hsm.core.actions import BasicActions

    received_args = {}

    def action_fn(arg1=None, arg2=None):
        nonlocal received_args
        received_args["arg1"] = arg1
        received_args["arg2"] = arg2

    BasicActions.execute(action_fn, arg1="test1", arg2="test2")
    assert received_args["arg1"] == "test1"
    assert received_args["arg2"] == "test2"
