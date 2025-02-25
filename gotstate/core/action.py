"""Action module for gotstate."""

from typing import Any, Callable, Dict, Optional
from gotstate.types.common import ActionId, EventId, EventData, StateAction


class Action:
    """
    Represents an action in the state machine.
    
    Actions are behaviors that are executed when:
    - A state is entered (entry action)
    - A state is exited (exit action)
    - A transition is taken (transition action)
    
    Actions can access the event data that triggered them.
    """
    
    def __init__(self, action_id: str, behavior: StateAction):
        """
        Initialize a new action.
        
        Args:
            action_id: Unique identifier for the action
            behavior: Function that implements the action behavior
        """
        self._action_id = ActionId(action_id)
        self._behavior = behavior
    
    @property
    def id(self) -> ActionId:
        """Get the action ID."""
        return self._action_id
    
    def execute(self, event_id: EventId, event_data: EventData) -> None:
        """
        Execute the action.
        
        Args:
            event_id: ID of the event that triggered the action
            event_data: Data associated with the event
        """
        self._behavior(event_id, event_data)
    
    def __eq__(self, other: Any) -> bool:
        """
        Compare equality with another action.
        
        Args:
            other: The other action to compare with
            
        Returns:
            True if the actions have the same ID, False otherwise
        """
        if not isinstance(other, Action):
            return False
        return self._action_id == other._action_id
    
    def __hash__(self) -> int:
        """
        Generate a hash for the action.
        
        Returns:
            Hash value for the action
        """
        return hash(self._action_id)
    
    def __repr__(self) -> str:
        """
        Generate a string representation of the action.
        
        Returns:
            String representation of the action
        """
        return f"Action(id={self._action_id})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the action to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the action
            
        Note:
            The behavior function cannot be serialized. When deserializing,
            the behavior must be re-supplied.
        """
        return {
            "action_id": self._action_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], behavior: StateAction) -> "Action":
        """
        Create an action from a dictionary.
        
        Args:
            data: Dictionary representation of the action
            behavior: Function that implements the action behavior
            
        Returns:
            New Action instance
        """
        return cls(action_id=data["action_id"], behavior=behavior) 