from threading import Event, Thread
from typing import Any, Callable


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
