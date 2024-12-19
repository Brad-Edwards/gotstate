# hsm/tests/utils.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import sys
from threading import Event, Thread
from typing import Any, Callable, Dict


class ThreadHelper:
    @staticmethod
    def run_with_timeout(target: Callable[[], Any], timeout: float = 0.2) -> Thread:
        """Run a function in a thread with timeout control"""
        thread = Thread(target=target)
        thread.start()
        thread.join(timeout=timeout)
        return thread

    @staticmethod
    def create_blocking_thread(ready_event: Event, release_event: Event, target: Callable[[], Any]) -> Thread:
        """Create a thread that blocks until signaled"""

        def wrapper():
            ready_event.set()
            target()
            release_event.set()

        thread = Thread(target=wrapper)
        thread.start()
        return thread


class MockDataStructures:
    @staticmethod
    def create_deep_dict(depth: int) -> dict:
        """Create a deeply nested dictionary"""
        deep_dict = {}
        current = deep_dict
        for i in range(depth):
            current["next"] = {}
            current = current["next"]
        return deep_dict

    @staticmethod
    def create_large_dict(size: int, value_size: int = 100) -> dict:
        """Create a large dictionary with controlled size"""
        return {str(i): "x" * value_size for i in range(size)}


"""
Common test utilities and data for HSM tests.
"""

# -----------------------------------------------------------------------------
# COMMON TEST DATA
# -----------------------------------------------------------------------------

PRIORITY_TEST_CASES = {
    "standard": [-100, -1, 0, 1, 100],
    "system_bounds": [
        sys.maxsize,
        -sys.maxsize - 1,
        sys.maxsize - 1,
        (-sys.maxsize - 1) + 1,
    ],
    "very_large": [
        2**1000,  # Extremely large number
        -(2**1000),  # Extremely large negative
        sys.maxsize * sys.maxsize,  # Product of max sizes
    ],
}

PAYLOAD_TEST_CASES = {
    "basic_types": [
        None,
        42,
        "string",
        ["list"],
        {"dict": "value"},
        set(),
        True,
        0.5,
    ],
    "empty_collections": [
        "",  # Empty string
        b"",  # Empty bytes
        [],  # Empty list
        (),  # Empty tuple
        set(),  # Empty set
        {},  # Empty dict
        frozenset(),  # Empty frozenset
        bytearray(),  # Empty bytearray
    ],
}

EVENT_ID_TEST_CASES = {
    "special": [
        "hello世界",  # Unicode
        "!@#$%^&*()",  # Special characters
        "\\n\\t\\r",  # Escape sequences
        "🌟🎉✨",  # Emojis
        " " * 100,  # Lots of spaces
        "\u0000\u0001\u0002",  # Control characters
    ],
    "invalid": [None, 42, True, [], {}],
}

# -----------------------------------------------------------------------------
# TEST HELPERS
# -----------------------------------------------------------------------------


def create_test_event(event_id: str = "test", payload: Any = None, priority: int = 0) -> "ValidEvent":
    """Helper function to create test events with default values."""
    return ValidEvent(event_id, payload, priority)


def create_large_data(depth: int = 100, width: int = 10000) -> Dict[str, Any]:
    """Helper function to create large test data structures."""
    large_data = {"deep_nest": {}, "large_list": list(range(width)), "large_string": "x" * 1000000}
    current = large_data["deep_nest"]
    for i in range(depth):
        current[f"level_{i}"] = {}
        current = current[f"level_{i}"]
    return large_data


# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS
# -----------------------------------------------------------------------------


class ValidEvent:
    """A valid event implementation for testing."""

    def __init__(self, event_id: str, payload: Any = None, priority: int = 0):
        if not isinstance(event_id, str):
            raise TypeError("event_id must be a string")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")

        self._id = event_id
        self._payload = payload
        self._priority = priority

    def get_id(self) -> str:
        return self._id

    def get_payload(self) -> Any:
        return self._payload

    def get_priority(self) -> int:
        return self._priority


class ValidTransition:
    """A valid transition implementation for testing."""

    def __init__(
        self,
        source: str,
        target: str,
        guard: Any = None,
        actions: Any = None,
        priority: int = 0,
    ) -> None:
        self._source = source
        self._target = target
        self._guard = guard
        self._actions = actions or []
        self._priority = priority

    def get_source_state_id(self) -> str:
        return self._source

    def get_target_state_id(self) -> str:
        return self._target

    def get_guard(self) -> Any:
        return self._guard

    def get_actions(self) -> list:
        return self._actions

    def get_priority(self) -> int:
        return self._priority
