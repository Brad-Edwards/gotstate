from hsm.interfaces.abc import AbstractTimer
from hsm.interfaces.protocols import Event


class Timer(AbstractTimer):
    def schedule_timeout(self, duration: float, event: "Event") -> None:
        raise NotImplementedError

    def cancel_timeout(self, event_id: str) -> None:
        raise NotImplementedError
