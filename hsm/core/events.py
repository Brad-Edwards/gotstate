from typing import Any

from hsm.interfaces.abc import AbstractEvent
from hsm.interfaces.types import EventID


class Event(AbstractEvent):
    def get_id(self) -> EventID:
        raise NotImplementedError

    def get_payload(self) -> Any:
        raise NotImplementedError

    def get_priority(self) -> int:
        raise NotImplementedError
