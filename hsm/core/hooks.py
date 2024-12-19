# hsm/core/hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import logging
from typing import Any, Callable, List, TypeVar

from hsm.core.errors import HSMError
from hsm.interfaces.abc import AbstractHook, AbstractTransition, StateID


class HookError(HSMError):
    """Raised when there's an error related to hook registration or management.

    Attributes:
        details: Additional context about the error.
    """

    pass


T = TypeVar("T")  # Type variable for hook method return type


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
        Register a hook. If the hook is already registered, this method does nothing.

        Runtime Invariants:
        - Registration is idempotent (registering same hook multiple times has no effect)
        - Hook identity is determined by object identity (is operator)

        Raises:
            HookError: If the object does not implement AbstractHook properly.
        """
        if not isinstance(hook, AbstractHook):
            raise HookError("Attempted to register a hook that does not implement AbstractHook.")
        if hook not in self._hooks:  # Use identity comparison
            self._hooks.append(hook)

    def unregister_hook(self, hook: AbstractHook) -> None:
        """
        Unregister a previously registered hook. Removes all instances of the hook.

        Runtime Invariants:
        - All instances of the hook are removed
        - Hook identity is determined by object identity (is operator)
        - If the hook is not found, this method does nothing
        """
        self._hooks = [h for h in self._hooks if h is not hook]

    def _call_hook_method(self, method_name: str, hook: AbstractHook, *args: Any) -> None:
        """Helper method to call a hook method and handle exceptions."""
        try:
            method = getattr(hook, method_name)
            method(*args)
        except Exception as e:
            self._logger.exception("Hook %s failed for args=%s: %s", method_name, args, e)

    def _call_hooks(self, method_name: str, *args: Any) -> None:
        """Helper method to call a specific method on all hooks."""
        for hook in self._hooks:
            self._call_hook_method(method_name, hook, *args)

    def call_on_enter(self, state_id: StateID) -> None:
        """
        Call on_enter(state_id) on all registered hooks.

        Hook failures are caught and logged.
        """
        self._call_hooks("on_enter", state_id)

    def call_on_exit(self, state_id: StateID) -> None:
        """
        Call on_exit(state_id) on all registered hooks.

        Hook failures are caught and logged.
        """
        self._call_hooks("on_exit", state_id)

    def call_pre_transition(self, transition: "AbstractTransition") -> None:
        """
        Call pre_transition(transition) on all registered hooks.

        Hook failures are caught and logged.
        """
        self._call_hooks("pre_transition", transition)

    def call_post_transition(self, transition: "AbstractTransition") -> None:
        """
        Call post_transition(transition) on all registered hooks.

        Hook failures are caught and logged.
        """
        self._call_hooks("post_transition", transition)
