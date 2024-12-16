# hsm/core/hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import logging
from typing import List

from hsm.core.errors import HSMError
from hsm.interfaces.abc import AbstractHook, AbstractTransition, StateID


class HookError(HSMError):
    """Raised when there's an error related to hook registration or management.

    Attributes:
        details: Additional context about the error.
    """

    pass


class HookManager:
    """
    Manages a list of hooks and invokes them at appropriate times.

    Runtime Invariants:
    - Hooks do not affect state machine behavior.
    - Hook failures are caught and logged, not raised.
    - Hooks are called in registration order.

    Example:
        manager = HookManager()
        manager.register_hook(some_hook)
        manager.call_on_enter("STATE_A")   # Calls on_enter on all registered hooks
    """

    def __init__(self) -> None:
        self._hooks: List[AbstractHook] = []
        self._logger = logging.getLogger("hsm.core.hooks")

    def register_hook(self, hook: AbstractHook) -> None:
        """
        Register a hook.

        Raises:
            HookError: If the object does not implement AbstractHook properly.
        """
        if not isinstance(hook, AbstractHook):
            raise HookError("Attempted to register a hook that does not implement AbstractHook.")
        self._hooks.append(hook)

    def unregister_hook(self, hook: AbstractHook) -> None:
        """
        Unregister a previously registered hook.

        If the hook is not found, this method does nothing.
        """
        try:
            self._hooks.remove(hook)
        except ValueError:
            # Hook not found, just ignore.
            pass

    def call_on_enter(self, state_id: StateID) -> None:
        """
        Call on_enter(state_id) on all registered hooks.

        Hook failures are caught and logged.
        """
        for hook in self._hooks:
            try:
                hook.on_enter(state_id)
            except Exception as e:
                self._logger.exception("Hook on_enter failed for state_id=%s: %s", state_id, e)

    def call_on_exit(self, state_id: StateID) -> None:
        """
        Call on_exit(state_id) on all registered hooks.

        Hook failures are caught and logged.
        """
        for hook in self._hooks:
            try:
                hook.on_exit(state_id)
            except Exception as e:
                self._logger.exception("Hook on_exit failed for state_id=%s: %s", state_id, e)

    def call_pre_transition(self, transition: "AbstractTransition") -> None:
        """
        Call pre_transition(transition) on all registered hooks.

        Hook failures are caught and logged.
        """
        for hook in self._hooks:
            try:
                hook.pre_transition(transition)
            except Exception as e:
                self._logger.exception("Hook pre_transition failed for transition=%s: %s", transition, e)

    def call_post_transition(self, transition: "AbstractTransition") -> None:
        """
        Call post_transition(transition) on all registered hooks.

        Hook failures are caught and logged.
        """
        for hook in self._hooks:
            try:
                hook.post_transition(transition)
            except Exception as e:
                self._logger.exception("Hook post_transition failed for transition=%s: %s", transition, e)
