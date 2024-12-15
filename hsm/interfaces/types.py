from typing import Any, Callable, Dict, NamedTuple, Protocol

StateID = str
EventID = str
Priority = int


class Event(Protocol):
    """Event protocol for type checking"""

    pass


class Transition(Protocol):
    """Transition protocol for type checking"""

    pass


class ValidationResult(NamedTuple):
    severity: str
    message: str
    context: Dict[str, Any]


TransitionCallback = Callable[..., None]
GuardCheck = Callable[[Any, Any], bool]
ActionExec = Callable[[Any, Any], None]
HookFunc = Callable[..., None]
