# hsm/core/states.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import Any, Dict

from hsm.interfaces.abc import AbstractCompositeState, AbstractState
from hsm.interfaces.types import StateID


class State(AbstractState):
    def on_enter(self) -> None:
        raise NotImplementedError

    def on_exit(self) -> None:
        raise NotImplementedError

    @property
    def data(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_id(self) -> StateID:
        raise NotImplementedError


class CompositeState(AbstractCompositeState, State):
    def get_substates(self) -> list[AbstractState]:
        raise NotImplementedError

    def get_initial_state(self) -> "State":
        raise NotImplementedError

    def has_history(self) -> bool:
        raise NotImplementedError
