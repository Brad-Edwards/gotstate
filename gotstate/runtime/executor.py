"""
Event execution and run-to-completion management.

Enforces run-to-completion semantics, manages transition execution,
and handles concurrent operations.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

import icontract

from gotstate.exceptions import GotStateError

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Defines the possible states of execution."""

    IDLE = auto()
    EXECUTING = auto()
    SUSPENDED = auto()
    ROLLING_BACK = auto()
    FAILED = auto()


class ExecutionMode(Enum):
    """Defines execution modes for the executor."""

    SYNCHRONOUS = auto()
    ASYNCHRONOUS = auto()
    PARALLEL = auto()


@icontract.invariant(lambda self: isinstance(self._status, ExecutionStatus), "Execution status must be valid")
class Executor:
    """Manages event execution with run-to-completion semantics.

    Class Invariants:
    1. Execution status must be a valid ExecutionStatus
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.SYNCHRONOUS) -> None:
        self._mode = mode
        self._status = ExecutionStatus.IDLE
        self._lock = threading.RLock()
        self._execution_count = 0

    @property
    def status(self) -> ExecutionStatus:
        return self._status

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def execution_count(self) -> int:
        return self._execution_count

    def execute(self, action: Callable[[], Any]) -> Any:
        """Execute an action with run-to-completion semantics."""
        with self._lock:
            if self._status == ExecutionStatus.EXECUTING:
                raise GotStateError("Cannot execute while another execution is in progress (RTC violation)")
            self._status = ExecutionStatus.EXECUTING
            try:
                result = action()
                self._execution_count += 1
                self._status = ExecutionStatus.IDLE
                return result
            except Exception:
                self._status = ExecutionStatus.FAILED
                raise

    def reset(self) -> None:
        """Reset the executor to IDLE state."""
        with self._lock:
            self._status = ExecutionStatus.IDLE


class ExecutionUnit:
    """Represents an atomic unit of execution with rollback capability."""

    def __init__(self, action: Callable[[], Any], rollback: Optional[Callable[[], None]] = None) -> None:
        self._action = action
        self._rollback = rollback
        self._executed = False
        self._result: Any = None

    @property
    def is_executed(self) -> bool:
        return self._executed

    @property
    def result(self) -> Any:
        return self._result

    def execute(self) -> Any:
        self._result = self._action()
        self._executed = True
        return self._result

    def undo(self) -> None:
        if self._executed and self._rollback is not None:
            self._rollback()
            self._executed = False


class ExecutionContext:
    """Maintains context for execution units."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
