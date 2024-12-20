from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(eq=False)
class StateBase:
    """Base class for state functionality"""

    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    parent: Optional["StateBase"] = None
    entry_actions: List[callable] = field(default_factory=list)
    exit_actions: List[callable] = field(default_factory=list)

    def __hash__(self) -> int:
        """Make states hashable based on their name and memory address."""
        # Use object id to break cycles while maintaining uniqueness
        return hash((self.name, id(self)))

    def __eq__(self, other: object) -> bool:
        """States are equal if they are the same object."""
        if not isinstance(other, StateBase):
            return NotImplemented
        return id(self) == id(other)

    def on_enter(self) -> None:
        """Execute entry actions"""
        for action in self.entry_actions:
            action()

    def on_exit(self) -> None:
        """Execute exit actions"""
        for action in self.exit_actions:
            action()
