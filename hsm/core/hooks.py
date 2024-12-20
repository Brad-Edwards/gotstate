# hsm/core/hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol

if TYPE_CHECKING:
    from hsm.core.states import State


class HookProtocol(Protocol):
    def on_enter(self, state: State) -> None: ...
    def on_exit(self, state: State) -> None: ...
    def on_error(self, error: Exception) -> None: ...


class HookManager:
    """
    Manages the registration and execution of hooks that listen to state machine
    lifecycle events (on_enter, on_exit, on_error). Users can attach logging,
    monitoring, or custom side effects without altering core logic.
    """

    def __init__(self, hooks: List["HookProtocol"] = None) -> None:
        """
        Initialize with an optional list of hook objects.
        :param hooks: A list of hook objects implementing HookProtocol.
        """
        self._hooks = hooks if hooks else []
        self._invoker = _HookInvoker(self._hooks)

    def register_hook(self, hook: "HookProtocol") -> None:
        """
        Add a new hook to the manager's list of hooks.

        :param hook: An object implementing HookProtocol methods.
        """
        self._hooks.append(hook)
        self._invoker = _HookInvoker(self._hooks)

    def execute_on_enter(self, state: "State") -> None:
        """
        Run all hooks' on_enter logic when entering a state.

        :param state: The state being entered.
        """
        self._invoker.invoke_on_enter(state)

    def execute_on_exit(self, state: "State") -> None:
        """
        Run all hooks' on_exit logic when exiting a state.

        :param state: The state being exited.
        """
        self._invoker.invoke_on_exit(state)

    def execute_on_error(self, error: Exception) -> None:
        """
        Run all hooks' on_error logic when an exception occurs.

        :param error: The exception encountered.
        """
        self._invoker.invoke_on_error(error)


class _HookInvoker:
    """
    Internal helper that iterates through a list of hooks and invokes their
    lifecycle methods in a controlled manner.
    """

    def __init__(self, hooks: List["HookProtocol"]) -> None:
        """
        Store hooks for invocation.
        :param hooks: A list of objects implementing HookProtocol methods.
        """
        self._hooks = hooks

    def invoke_on_enter(self, state: "State") -> None:
        """
        Call each hook's on_enter method.

        :param state: The state being entered.
        """
        for hook in self._hooks:
            if hasattr(hook, "on_enter"):
                hook.on_enter(state)

    def invoke_on_exit(self, state: "State") -> None:
        """
        Call each hook's on_exit method.

        :param state: The state being exited.
        """
        for hook in self._hooks:
            if hasattr(hook, "on_exit"):
                hook.on_exit(state)

    def invoke_on_error(self, error: Exception) -> None:
        """
        Call each hook's on_error method.

        :param error: The exception that occurred.
        """
        for hook in self._hooks:
            if hasattr(hook, "on_error"):
                hook.on_error(error)


class Hook:
    def __init__(self, callback: Callable[..., Any], priority: Optional[int] = None):
        self.callback = callback
        self.priority = priority if priority is not None else 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.callback(*args, **kwargs)
