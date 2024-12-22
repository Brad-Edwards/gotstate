# hsm/runtime/state_history.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from threading import Lock
from typing import Dict, Optional

from hsm.core.states import CompositeState, State


class StateHistory:
    """Thread-safe state history management."""

    def __init__(self):
        self._history: Dict[CompositeState, State] = {}
        self._lock = Lock()

    def record_state(self, composite_state: CompositeState, last_active_state: State) -> None:
        """Thread-safe recording of state history."""
        with self._lock:
            self._history[composite_state] = last_active_state

    def get_last_state(self, composite_state: CompositeState) -> Optional[State]:
        """Thread-safe retrieval of last active state."""
        with self._lock:
            return self._history.get(composite_state)

    def clear(self) -> None:
        """Thread-safe clearing of history."""
        with self._lock:
            self._history.clear()
