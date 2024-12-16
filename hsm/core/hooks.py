# hsm/core/hooks.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from hsm.interfaces.abc import AbstractHook
from hsm.interfaces.protocols import Transition


class DefaultHook(AbstractHook):
    def on_enter(self, state_id: str) -> None:
        raise NotImplementedError

    def on_exit(self, state_id: str) -> None:
        raise NotImplementedError

    def pre_transition(self, transition: "Transition") -> None:
        raise NotImplementedError

    def post_transition(self, transition: "Transition") -> None:
        raise NotImplementedError
