# hsm/core/events.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
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
