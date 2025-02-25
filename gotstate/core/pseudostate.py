"""PseudoState module for gotstate."""

from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, TypeVar, cast

from gotstate.types.common import StateId
from gotstate.core.exceptions import PseudoStateError


class PseudoStateKind(Enum):
    """Enumeration of the different kinds of pseudostates in UML."""
    
    INITIAL = auto()
    DEEP_HISTORY = auto()
    SHALLOW_HISTORY = auto()
    JOIN = auto()
    FORK = auto()
    JUNCTION = auto()
    CHOICE = auto()
    ENTRY_POINT = auto()
    EXIT_POINT = auto()
    TERMINATE = auto()


class PseudoState:
    """
    Base class for all pseudostates in a state machine.
    
    Pseudostates are transient vertices in a state machine that are used to
    construct complex transition paths.
    """
    
    def __init__(self, name: str, kind: PseudoStateKind, parent: Optional["State"] = None):
        """
        Initialize a new pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            kind: Kind of pseudostate
            parent: Optional parent state that contains this pseudostate
        """
        from gotstate.core.state import State
        
        self._state_id = StateId(name)
        self._kind = kind
        self._parent = parent
        
        # Add to parent's children if parent exists
        if parent is not None:
            parent._children.add(self)
    
    @property
    def id(self) -> StateId:
        """Get the pseudostate ID."""
        return self._state_id
    
    @property
    def name(self) -> str:
        """Get the pseudostate name."""
        return str(self._state_id)
    
    @property
    def kind(self) -> PseudoStateKind:
        """Get the kind of pseudostate."""
        return self._kind
    
    @property
    def parent(self) -> Optional["State"]:
        """Get the parent state."""
        return self._parent
    
    def __eq__(self, other: Any) -> bool:
        """
        Compare equality with another pseudostate.
        
        Args:
            other: The other pseudostate to compare with
            
        Returns:
            True if the pseudostates have the same ID and kind, False otherwise
        """
        if not isinstance(other, PseudoState):
            return False
        return self._state_id == other._state_id and self._kind == other._kind
    
    def __hash__(self) -> int:
        """
        Generate a hash for the pseudostate.
        
        Returns:
            Hash value for the pseudostate
        """
        return hash((self._state_id, self._kind))
    
    def __repr__(self) -> str:
        """
        Generate a string representation of the pseudostate.
        
        Returns:
            String representation of the pseudostate
        """
        return f"PseudoState(id={self._state_id}, kind={self._kind.name})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pseudostate to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the pseudostate
        """
        return {
            "state_id": self._state_id,
            "kind": self._kind.name,
            "parent_id": self._parent.id if self._parent else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional["State"] = None) -> "PseudoState":
        """
        Create a pseudostate from a dictionary.
        
        Args:
            data: Dictionary representation of the pseudostate
            parent: Optional parent state
            
        Returns:
            New PseudoState instance
        """
        kind = PseudoStateKind[data["kind"]]
        return cls(data["state_id"], kind, parent)


class InitialState(PseudoState):
    """
    Represents an initial pseudostate.
    
    An initial pseudostate represents a default vertex that is the source
    for a single transition to the default state of a composite state.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new initial pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.INITIAL, parent)


class HistoryState(PseudoState):
    """
    Base class for history pseudostates.
    
    A history pseudostate represents a remembrance of the last active substate
    of a composite state.
    """
    
    def __init__(self, name: str, kind: PseudoStateKind, parent: Optional["State"] = None, default_target: Optional["State"] = None):
        """
        Initialize a new history pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            kind: Kind of history pseudostate (deep or shallow)
            parent: Optional parent state that contains this pseudostate
            default_target: Optional default target state if history is not available
        """
        if kind not in (PseudoStateKind.DEEP_HISTORY, PseudoStateKind.SHALLOW_HISTORY):
            raise PseudoStateError(f"History state must be of kind DEEP_HISTORY or SHALLOW_HISTORY, got {kind}")
        
        super().__init__(name, kind, parent)
        self._default_target = default_target
        self._history: Optional["State"] = None
    
    @property
    def default_target(self) -> Optional["State"]:
        """Get the default target state."""
        return self._default_target
    
    @property
    def history(self) -> Optional["State"]:
        """Get the remembered state."""
        return self._history
    
    def set_history(self, state: "State") -> None:
        """
        Set the remembered state.
        
        Args:
            state: The state to remember
        """
        self._history = state
    
    def reset(self) -> None:
        """Reset the history, forgetting any remembered state."""
        self._history = None


class DeepHistoryState(HistoryState):
    """
    Represents a deep history pseudostate.
    
    A deep history pseudostate represents a remembrance of the last active
    substates at all levels within a composite state.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None, default_target: Optional["State"] = None):
        """
        Initialize a new deep history pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
            default_target: Optional default target state if history is not available
        """
        super().__init__(name, PseudoStateKind.DEEP_HISTORY, parent, default_target)


class ShallowHistoryState(HistoryState):
    """
    Represents a shallow history pseudostate.
    
    A shallow history pseudostate represents a remembrance of only the last active
    direct substate of a composite state, not including deeper levels.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None, default_target: Optional["State"] = None):
        """
        Initialize a new shallow history pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
            default_target: Optional default target state if history is not available
        """
        super().__init__(name, PseudoStateKind.SHALLOW_HISTORY, parent, default_target)


class JunctionState(PseudoState):
    """
    Represents a junction pseudostate.
    
    A junction pseudostate is used to chain together multiple transitions. It is used to
    construct complex transition paths where a single trigger can result in different
    target states based on multiple guard conditions.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new junction pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.JUNCTION, parent)


class ChoiceState(PseudoState):
    """
    Represents a choice pseudostate.
    
    A choice pseudostate is similar to a junction, but it is used when the guard
    conditions need to be evaluated at the time the transition is executed, rather
    than when the transition is triggered.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new choice pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.CHOICE, parent)


class ForkState(PseudoState):
    """
    Represents a fork pseudostate.
    
    A fork pseudostate is used to split an incoming transition into multiple
    outgoing transitions targeting states in different orthogonal regions.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new fork pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.FORK, parent)


class JoinState(PseudoState):
    """
    Represents a join pseudostate.
    
    A join pseudostate is used to merge several transitions coming from source
    states in different orthogonal regions. The outgoing transition from a join
    pseudostate can have a guard condition.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new join pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.JOIN, parent)


class EntryPointState(PseudoState):
    """
    Represents an entry point pseudostate.
    
    An entry point pseudostate is an entry point to a state machine or composite state.
    It provides an explicit entry point and additional semantics for entering a state machine
    or composite state.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new entry point pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.ENTRY_POINT, parent)


class ExitPointState(PseudoState):
    """
    Represents an exit point pseudostate.
    
    An exit point pseudostate is an exit point from a state machine or composite state.
    It provides an explicit exit point and additional semantics for exiting a state machine
    or composite state.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new exit point pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.EXIT_POINT, parent)


class TerminateState(PseudoState):
    """
    Represents a terminate pseudostate.
    
    A terminate pseudostate implies that the execution of the state machine
    is terminated immediately.
    """
    
    def __init__(self, name: str, parent: Optional["State"] = None):
        """
        Initialize a new terminate pseudostate.
        
        Args:
            name: Name of the pseudostate, used as its identifier
            parent: Optional parent state that contains this pseudostate
        """
        super().__init__(name, PseudoStateKind.TERMINATE, parent) 