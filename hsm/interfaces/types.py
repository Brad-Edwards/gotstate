# hsm/interfaces/types.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
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
