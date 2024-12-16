# hsm/core/actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import Any

from hsm.interfaces.abc import AbstractAction
from hsm.interfaces.protocols import Event


class BasicAction(AbstractAction):
    def execute(self, event: "Event", state_data: Any) -> None:
        raise NotImplementedError
