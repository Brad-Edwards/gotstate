from typing import List, Optional

from hsm.interfaces.abc import AbstractAction, AbstractGuard, AbstractTransition
from hsm.interfaces.types import EventID, StateID


class Transition(AbstractTransition):
    def get_source_state_id(self) -> StateID:
        raise NotImplementedError

    def get_target_state_id(self) -> StateID:
        raise NotImplementedError

    def get_guard(self) -> Optional[AbstractGuard]:
        raise NotImplementedError

    def get_actions(self) -> List[AbstractAction]:
        raise NotImplementedError

    def get_priority(self) -> int:
        raise NotImplementedError
