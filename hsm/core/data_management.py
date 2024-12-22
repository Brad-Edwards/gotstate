# hsm/core/data_management.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator

from hsm.core.states import State
from hsm.runtime.graph import StateGraph


class _DataLockManager:
    """
    Internal lock manager controlling access to state data. Ensures thread-safe
    reads and writes when using direct state access (legacy support).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def lock(self) -> None:
        """Acquire the data lock."""
        self._lock.acquire()

    def unlock(self) -> None:
        """Release the data lock."""
        self._lock.release()


class _ScopedDataContext:
    """
    Internal context manager for legacy direct state access.
    New code should use state_data_context with graph instead.
    """

    def __init__(self, lock_manager: _DataLockManager) -> None:
        self._lock_manager = lock_manager

    def __enter__(self) -> None:
        self._lock_manager.lock()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._lock_manager.unlock()


def with_state_data_lock(state: State):
    """
    Legacy support for direct state data access.
    New code should use state_data_context with graph instead.
    """
    if not hasattr(state, "_lock_manager"):
        state._lock_manager = _DataLockManager()
    return _ScopedDataContext(state._lock_manager)


@contextmanager
def state_data_context(graph: StateGraph, state: State) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for safely accessing state data through the graph.
    This is the preferred way to access state data.

    Example:
        with state_data_context(graph, state) as data:
            data["key"] = value
    """
    with graph._data_locks[state]:
        data = graph.get_state_data(state)
        yield data


def get_state_data(graph: StateGraph, state: State, key: str, default: Any = None) -> Any:
    """Thread-safe way to get a specific piece of state data through the graph."""
    with graph._data_locks[state]:
        return graph.get_state_data(state).get(key, default)


def set_state_data(graph: StateGraph, state: State, key: str, value: Any) -> None:
    """Thread-safe way to set a specific piece of state data through the graph."""
    graph.set_state_data(state, key, value)
