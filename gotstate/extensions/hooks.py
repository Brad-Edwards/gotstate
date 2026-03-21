"""
Extension interface and lifecycle management.

Defines extension hook interfaces, manages extension lifecycle,
and provides customization points for state machine behavior.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import icontract


class HookPhase(Enum):
    """Defines extension hook execution phases."""

    PRE = auto()
    MAIN = auto()
    POST = auto()
    ERROR = auto()
    CLEANUP = auto()


class HookPriority(Enum):
    """Defines hook execution priorities."""

    HIGHEST = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    LOWEST = 4


class ExtensionHooks(ABC):
    """Defines extension hook interfaces.

    Provides extensible behavior through hook points.
    """

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    def __init__(self, name: str, priority: HookPriority = HookPriority.NORMAL) -> None:
        self._name = name
        self._priority = priority
        self._active = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> HookPriority:
        return self._priority

    @property
    def is_active(self) -> bool:
        return self._active

    @abstractmethod
    def on_pre(self, context: Dict[str, Any]) -> None:
        """Called before the main operation."""
        ...

    @abstractmethod
    def on_post(self, context: Dict[str, Any]) -> None:
        """Called after the main operation."""
        ...

    @abstractmethod
    def on_error(self, context: Dict[str, Any], error: Exception) -> None:
        """Called when an error occurs."""
        ...

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False


class HookManager:
    """Manages extension hook lifecycle and execution."""

    def __init__(self) -> None:
        self._hooks: Dict[str, List[ExtensionHooks]] = {}
        self._lock = threading.Lock()

    def register(self, phase: str, hook: ExtensionHooks) -> None:
        """Register a hook for a given phase."""
        with self._lock:
            if phase not in self._hooks:
                self._hooks[phase] = []
            self._hooks[phase].append(hook)
            self._hooks[phase].sort(key=lambda h: h.priority.value)
            hook.activate()

    def unregister(self, phase: str, hook_name: str) -> None:
        with self._lock:
            if phase in self._hooks:
                self._hooks[phase] = [h for h in self._hooks[phase] if h.name != hook_name]

    def execute_hooks(self, phase: str, context: Dict[str, Any]) -> None:
        """Execute all hooks for a given phase in priority order."""
        with self._lock:
            hooks = list(self._hooks.get(phase, []))
        for hook in hooks:
            if hook.is_active:
                try:
                    if phase.endswith("_pre"):
                        hook.on_pre(context)
                    elif phase.endswith("_post"):
                        hook.on_post(context)
                except Exception as e:
                    hook.on_error(context, e)

    def get_hooks(self, phase: str) -> List[ExtensionHooks]:
        with self._lock:
            return list(self._hooks.get(phase, []))
