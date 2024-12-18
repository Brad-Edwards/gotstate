import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

T = TypeVar("T")


class AsyncTestHelper:
    @staticmethod
    async def run_with_timeout(coro: Awaitable[T], timeout: float = 0.2) -> T:
        """Run a coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

    @staticmethod
    async def create_blocking_task(
        ready_event: asyncio.Event, release_event: asyncio.Event, coro: Callable[[], Awaitable[T]]
    ) -> asyncio.Task:
        """Create a task that blocks until signaled"""

        async def wrapper():
            ready_event.set()
            await coro()
            release_event.set()

        return asyncio.create_task(wrapper())


class AsyncMock:
    """Base class for async mocks"""

    def __init__(self):
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.return_value

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self):
        self.calls = []
