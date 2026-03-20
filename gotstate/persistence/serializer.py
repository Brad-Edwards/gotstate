"""
State machine serialization and persistence management.

Handles state machine persistence in multiple formats with
version compatibility and migration support.
"""

from __future__ import annotations

import json
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import icontract

from gotstate.exceptions import SerializationError

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Defines supported serialization formats."""

    JSON = auto()
    BINARY = auto()
    XML = auto()
    YAML = auto()


class VersionCompatibility(Enum):
    """Defines version compatibility levels."""

    EXACT = auto()
    COMPATIBLE = auto()
    MIGRATION = auto()
    BREAKING = auto()


@icontract.invariant(lambda self: isinstance(self._format, SerializationFormat), "Serialization format must be valid")
class Serializer:
    """Manages state machine serialization and persistence.

    Class Invariants:
    1. Serialization format must be a valid SerializationFormat
    """

    def __init__(self, fmt: SerializationFormat = SerializationFormat.JSON) -> None:
        self._format = fmt
        self._version = "1.0.0"

    @property
    def format(self) -> SerializationFormat:
        return self._format

    @property
    def version(self) -> str:
        return self._version

    def serialize(self, machine: Any) -> str:
        """Serialize a state machine to string representation."""
        from gotstate.core.machine import StateMachine

        if not isinstance(machine, StateMachine):
            raise SerializationError("Can only serialize StateMachine instances")

        if self._format == SerializationFormat.JSON:
            return self._serialize_json(machine)
        raise SerializationError(f"Unsupported format: {self._format}")

    def _serialize_json(self, machine: Any) -> str:
        data = {
            "version": self._version,
            "name": machine.name,
            "status": machine.status.name,
            "states": [{"name": s.name, "type": s.state_type.name} for s in machine.states.values()],
            "transitions": [
                {
                    "source": t.source.name,
                    "target": t.target.name if t.target else None,
                    "kind": t.kind.name,
                    "trigger": t.trigger,
                }
                for t in machine.transitions
            ],
            "current_state": machine.current_state.name if machine.current_state else None,
        }
        return json.dumps(data, indent=2)

    def deserialize(self, data: str) -> Dict[str, Any]:
        """Deserialize a string to a machine definition dict."""
        if self._format == SerializationFormat.JSON:
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                raise SerializationError(f"Invalid JSON: {e}") from e
        raise SerializationError(f"Unsupported format: {self._format}")
