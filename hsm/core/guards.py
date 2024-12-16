from typing import Any

from hsm.interfaces.abc import AbstractGuard
from hsm.interfaces.protocols import Event


class BasicGuard(AbstractGuard):
    def check(self, event: "Event", state_data: Any) -> bool:
        raise NotImplementedError
