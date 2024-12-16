from typing import Any, Callable, Dict, NamedTuple

StateID = str
EventID = str
Priority = int


class ValidationResult(NamedTuple):
    severity: str
    message: str
    context: Dict[str, Any]


# Callback Types
TransitionCallback = Callable[..., None]
GuardCheck = Callable[[Any, Any], bool]
ActionExec = Callable[[Any, Any], None]
HookFunc = Callable[..., None]
