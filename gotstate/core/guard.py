"""Guard module for gotstate."""

from typing import Any, Callable, Dict, Optional
from gotstate.types.common import GuardId, EventId, EventData, GuardFunction


class Guard:
    """
    Represents a guard condition in the state machine.
    
    Guards are boolean conditions that determine whether a transition can be taken.
    They are evaluated when an event is processed and can access the event data.
    """
    
    def __init__(self, guard_id: str, condition: GuardFunction):
        """
        Initialize a new guard.
        
        Args:
            guard_id: Unique identifier for the guard
            condition: Function that evaluates the guard condition
        """
        self._guard_id = GuardId(guard_id)
        self._condition = condition
    
    @property
    def id(self) -> GuardId:
        """Get the guard ID."""
        return self._guard_id
    
    def evaluate(self, event_id: EventId, event_data: EventData) -> bool:
        """
        Evaluate the guard condition.
        
        Args:
            event_id: ID of the event being processed
            event_data: Data associated with the event
            
        Returns:
            True if the guard condition is satisfied, False otherwise
        """
        return self._condition(event_id, event_data)
    
    def __eq__(self, other: Any) -> bool:
        """
        Compare equality with another guard.
        
        Args:
            other: The other guard to compare with
            
        Returns:
            True if the guards have the same ID, False otherwise
        """
        if not isinstance(other, Guard):
            return False
        return self._guard_id == other._guard_id
    
    def __hash__(self) -> int:
        """
        Generate a hash for the guard.
        
        Returns:
            Hash value for the guard
        """
        return hash(self._guard_id)
    
    def __repr__(self) -> str:
        """
        Generate a string representation of the guard.
        
        Returns:
            String representation of the guard
        """
        return f"Guard(id={self._guard_id})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the guard to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the guard
            
        Note:
            The condition function cannot be serialized. When deserializing,
            the condition must be re-supplied.
        """
        return {
            "guard_id": self._guard_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], condition: GuardFunction) -> "Guard":
        """
        Create a guard from a dictionary.
        
        Args:
            data: Dictionary representation of the guard
            condition: Function that evaluates the guard condition
            
        Returns:
            New Guard instance
        """
        return cls(guard_id=data["guard_id"], condition=condition) 