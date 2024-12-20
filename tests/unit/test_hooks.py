# tests/unit/test_hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from typing import List

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
