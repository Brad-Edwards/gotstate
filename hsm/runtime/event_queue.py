from hsm.interfaces.abc import AbstractEvent, AbstractEventQueue
from hsm.interfaces.types import Event


class EventQueue(AbstractEventQueue):
    def enqueue(self, event: "Event") -> None:
        raise NotImplementedError

    def dequeue(self) -> AbstractEvent:
        raise NotImplementedError

    def is_full(self) -> bool:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError
