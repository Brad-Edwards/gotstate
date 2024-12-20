# tests/unit/test_hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import List
from unittest.mock import MagicMock

import pytest

from hsm.core.hooks import Hook
from hsm.core.states import State


def test_hook_manager(dummy_hooks: List[Hook], dummy_state: State):
    from hsm.core.hooks import HookManager

    hm = HookManager(hooks=dummy_hooks)
    hm.execute_on_enter(dummy_state)
    dummy_hooks[0].on_enter.assert_called_once_with(dummy_state)
    hm.execute_on_exit(dummy_state)
    dummy_hooks[0].on_exit.assert_called_once_with(dummy_state)
    err = Exception("TestError")
    hm.execute_on_error(err)
    dummy_hooks[0].on_error.assert_called_once_with(err)


def test_hook_manager_register(dummy_hooks: List[Hook]):
    from hsm.core.hooks import HookManager

    hm = HookManager()
    hm.register_hook(dummy_hooks[0])
    # Now hook is included in manager's list
    assert len(hm._hooks) == 1


def test_hook_manager_init():
    from hsm.core.hooks import HookManager, HookProtocol

    # Test empty initialization
    hm = HookManager()
    assert len(hm._hooks) == 0

    # Test with hooks
    mock_hook = MagicMock(spec=HookProtocol)
    hm = HookManager(hooks=[mock_hook])
    assert len(hm._hooks) == 1


def test_hook_manager_register():
    from hsm.core.hooks import HookManager, HookProtocol

    hm = HookManager()
    mock_hook = MagicMock(spec=HookProtocol)
    hm.register_hook(mock_hook)
    assert len(hm._hooks) == 1
    assert hm._hooks[0] == mock_hook


def test_hook_invoker():
    from hsm.core.hooks import HookProtocol, _HookInvoker
    from hsm.core.states import State

    mock_hook = MagicMock(spec=HookProtocol)
    invoker = _HookInvoker([mock_hook])

    state = State("test")
    error = Exception("test error")

    invoker.invoke_on_enter(state)
    mock_hook.on_enter.assert_called_once_with(state)

    invoker.invoke_on_exit(state)
    mock_hook.on_exit.assert_called_once_with(state)

    invoker.invoke_on_error(error)
    mock_hook.on_error.assert_called_once_with(error)


def test_hook_invoker_missing_methods():
    from hsm.core.hooks import _HookInvoker
    from hsm.core.states import State

    # Create a mock hook without all methods
    incomplete_hook = MagicMock()
    del incomplete_hook.on_enter  # Remove on_enter method

    invoker = _HookInvoker([incomplete_hook])
    state = State("test")
    error = Exception("test error")

    # Should not raise errors when methods are missing
    invoker.invoke_on_enter(state)
    invoker.invoke_on_exit(state)
    invoker.invoke_on_error(error)


def test_hook_class():
    from hsm.core.hooks import Hook

    callback_called = False

    def test_callback():
        nonlocal callback_called
        callback_called = True

    # Test with default priority
    hook = Hook(test_callback)
    assert hook.priority == 0

    # Test with custom priority
    hook = Hook(test_callback, priority=1)
    assert hook.priority == 1

    # Test callback execution
    hook()
    assert callback_called is True


def test_hook_with_arguments():
    from hsm.core.hooks import Hook

    received_args = None
    received_kwargs = None

    def test_callback(*args, **kwargs):
        nonlocal received_args, received_kwargs
        received_args = args
        received_kwargs = kwargs

    hook = Hook(test_callback)
    hook(1, 2, key="value")
    assert received_args == (1, 2)
    assert received_kwargs == {"key": "value"}


def test_hook_manager_execution_order():
    from hsm.core.hooks import HookManager, HookProtocol
    from hsm.core.states import State

    execution_order = []

    class OrderedHook(HookProtocol):
        def __init__(self, order):
            self.order = order

        def on_enter(self, state):
            execution_order.append(self.order)

        def on_exit(self, state):
            pass

        def on_error(self, error):
            pass

    hook1 = OrderedHook(1)
    hook2 = OrderedHook(2)

    hm = HookManager([hook1, hook2])
    state = State("test")
    hm.execute_on_enter(state)

    assert execution_order == [1, 2]
