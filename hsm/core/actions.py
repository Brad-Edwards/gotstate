from typing import Any

from hsm.interfaces.abc import AbstractAction
from hsm.interfaces.types import Event


class BasicAction(AbstractAction):
    def execute(self, event: "Event", state_data: Any) -> None:
        raise NotImplementedError
