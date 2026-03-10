"""
Extension isolation and security management.

Implements extension isolation, enforces resource boundaries,
and manages extension security.
"""

from __future__ import annotations

import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Set

import icontract

from gotstate.exceptions import GotStateError


class SecurityLevel(Enum):
    """Defines security enforcement levels."""

    STRICT = auto()
    HIGH = auto()
    STANDARD = auto()
    RELAXED = auto()


class ResourceLimit(Enum):
    """Defines resource limitation types."""

    MEMORY = auto()
    CPU = auto()
    IO = auto()
    NETWORK = auto()
    STORAGE = auto()


@icontract.invariant(lambda self: isinstance(self._security_level, SecurityLevel), "Security level must be valid")
class ExtensionSandbox:
    """Manages extension isolation and security.

    Provides secure execution environments for extensions
    with resource limits and access control.

    Class Invariants:
    1. Security level must be a valid SecurityLevel
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD) -> None:
        self._security_level = security_level
        self._resource_limits: Dict[ResourceLimit, float] = {}
        self._resource_usage: Dict[ResourceLimit, float] = {}
        self._allowed_operations: Set[str] = set()
        self._violations: list = []
        self._lock = threading.Lock()

    @property
    def security_level(self) -> SecurityLevel:
        return self._security_level

    @property
    def violations(self) -> list:
        return list(self._violations)

    def set_resource_limit(self, resource: ResourceLimit, limit: float) -> None:
        """Set a resource usage limit."""
        with self._lock:
            self._resource_limits[resource] = limit
            self._resource_usage.setdefault(resource, 0.0)

    def check_resource(self, resource: ResourceLimit, amount: float) -> bool:
        """Check if resource usage is within limits."""
        with self._lock:
            limit = self._resource_limits.get(resource)
            if limit is None:
                return True
            current = self._resource_usage.get(resource, 0.0)
            return current + amount <= limit

    def record_usage(self, resource: ResourceLimit, amount: float) -> None:
        """Record resource usage."""
        with self._lock:
            if not self.check_resource(resource, amount):
                self._violations.append(
                    {"resource": resource.name, "amount": amount, "timestamp": time.monotonic()}
                )
                raise GotStateError(f"Resource limit exceeded for {resource.name}")
            self._resource_usage[resource] = self._resource_usage.get(resource, 0.0) + amount

    def allow_operation(self, operation: str) -> None:
        """Allow a specific operation in the sandbox."""
        self._allowed_operations.add(operation)

    def is_operation_allowed(self, operation: str) -> bool:
        """Check if an operation is allowed."""
        if self._security_level == SecurityLevel.RELAXED:
            return True
        return operation in self._allowed_operations

    def execute_sandboxed(self, action: Callable[[], Any]) -> Any:
        """Execute an action within the sandbox constraints."""
        try:
            return action()
        except Exception as e:
            self._violations.append(
                {"type": "execution_error", "error": str(e), "timestamp": time.monotonic()}
            )
            raise

    def reset_usage(self) -> None:
        """Reset resource usage counters."""
        with self._lock:
            self._resource_usage.clear()
