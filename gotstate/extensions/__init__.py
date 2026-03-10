"""
Extensions package for extension mechanisms.

Defines extension interfaces, manages extension lifecycle,
provides extension points, and enforces sandbox security.
"""

from .hooks import ExtensionHooks, HookManager, HookPhase, HookPriority
from .sandbox import ExtensionSandbox, ResourceLimit, SecurityLevel

__all__ = [
    "ExtensionHooks",
    "HookPhase",
    "HookPriority",
    "HookManager",
    "ExtensionSandbox",
    "SecurityLevel",
    "ResourceLimit",
]
