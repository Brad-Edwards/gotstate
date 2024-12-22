# tests/unit/test_guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details


def test_basic_actions_execute():
    from gotstate.core.actions import BasicActions

    action_called = False

    def action_fn(arg=None):
        nonlocal action_called
        action_called = True
        assert arg == "test"

    BasicActions.execute(action_fn, arg="test")
    assert action_called is True


def test_basic_guards_check_condition():
    from gotstate.core.guards import BasicGuards

    condition_called = False

    def condition_fn(arg=None):
        nonlocal condition_called
        condition_called = True
        return arg == "test"

    result = BasicGuards.check_condition(condition_fn, arg="test")
    assert result is True
    assert condition_called is True


def test_basic_guards_check_condition_false():
    from gotstate.core.guards import BasicGuards

    def condition_fn(arg=None):
        return arg == "test"

    result = BasicGuards.check_condition(condition_fn, arg="wrong")
    assert result is False


def test_basic_guards_check_condition_no_args():
    from gotstate.core.guards import BasicGuards

    def condition_fn():
        return True

    result = BasicGuards.check_condition(condition_fn)
    assert result is True


def test_guard_adapter_init():
    from gotstate.core.guards import _GuardAdapter

    def guard_fn(event):
        return True

    adapter = _GuardAdapter(guard_fn)
    assert adapter._guard_fn == guard_fn


def test_guard_adapter_check():
    from gotstate.core.events import Event
    from gotstate.core.guards import _GuardAdapter

    event_received = None

    def guard_fn(event):
        nonlocal event_received
        event_received = event
        return True

    adapter = _GuardAdapter(guard_fn)
    test_event = Event("TestEvent")
    result = adapter.check(test_event)

    assert result is True
    assert event_received == test_event


def test_guard_adapter_check_false():
    from gotstate.core.events import Event
    from gotstate.core.guards import _GuardAdapter

    def guard_fn(event):
        return False

    adapter = _GuardAdapter(guard_fn)
    test_event = Event("TestEvent")
    result = adapter.check(test_event)

    assert result is False
