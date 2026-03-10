"""
Type system extension management.

Provides type extension points, manages type conversions,
and maintains type consistency for extension-provided types.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

import icontract

from gotstate.types.base import BaseType, TypeKind


class ExtensionStatus(Enum):
    """Defines extension lifecycle states."""

    UNREGISTERED = auto()
    REGISTERING = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    UNLOADING = auto()


class ExtensionScope(Enum):
    """Defines extension visibility scopes."""

    PRIVATE = auto()
    SHARED = auto()
    PUBLIC = auto()
    SYSTEM = auto()


class TypeExtension(ABC):
    """Base class for type system extensions.

    Provides extensible type system functionality via the Plugin pattern.
    """

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    def __init__(self, name: str, scope: ExtensionScope = ExtensionScope.PRIVATE) -> None:
        self._name = name
        self._scope = scope
        self._status = ExtensionStatus.UNREGISTERED

    @property
    def name(self) -> str:
        return self._name

    @property
    def scope(self) -> ExtensionScope:
        return self._scope

    @property
    def status(self) -> ExtensionStatus:
        return self._status

    @abstractmethod
    def get_types(self) -> List[BaseType]:
        """Return all types provided by this extension."""
        ...

    @abstractmethod
    def convert(self, value: Any, target_type: BaseType) -> Any:
        """Convert a value to the target type."""
        ...

    def activate(self) -> None:
        self._status = ExtensionStatus.ACTIVE

    def deactivate(self) -> None:
        self._status = ExtensionStatus.SUSPENDED


class ExtensionManager:
    """Manages type system extensions and their lifecycle."""

    def __init__(self) -> None:
        self._extensions: Dict[str, TypeExtension] = {}
        self._lock = threading.Lock()

    def register(self, extension: TypeExtension) -> None:
        with self._lock:
            extension._status = ExtensionStatus.REGISTERING
            self._extensions[extension.name] = extension
            extension.activate()

    def unregister(self, name: str) -> None:
        with self._lock:
            if name in self._extensions:
                ext = self._extensions.pop(name)
                ext._status = ExtensionStatus.UNLOADING

    def get(self, name: str) -> Optional[TypeExtension]:
        with self._lock:
            return self._extensions.get(name)

    def get_all_types(self) -> List[BaseType]:
        """Return all types from all active extensions."""
        types: List[BaseType] = []
        with self._lock:
            for ext in self._extensions.values():
                if ext.status == ExtensionStatus.ACTIVE:
                    types.extend(ext.get_types())
        return types
