"""Exception classes for gotstate."""

class GotStateError(Exception):
    """Base exception for all gotstate errors."""
    pass


class StateError(GotStateError):
    """Exception raised for errors related to states."""
    pass


class StateNotFoundError(StateError):
    """Exception raised when a state is not found."""
    pass


class DuplicateStateError(StateError):
    """Exception raised when attempting to add a duplicate state."""
    pass


class InvalidStateError(StateError):
    """Exception raised when a state is invalid for the current operation."""
    pass


class TransitionError(GotStateError):
    """Exception raised for errors related to transitions."""
    pass


class InvalidTransitionError(TransitionError):
    """Exception raised when a transition is invalid for the current state."""
    pass


class TransitionNotFoundError(TransitionError):
    """Exception raised when a transition is not found."""
    pass


class EventError(GotStateError):
    """Exception raised for errors related to events."""
    pass


class EventNotAcceptedError(EventError):
    """Exception raised when an event is not accepted by the current state."""
    pass


class InvalidEventError(EventError):
    """Exception raised when an event is invalid."""
    pass


class StateMachineError(GotStateError):
    """Exception raised for errors related to the state machine."""
    pass


class StateMachineNotStartedError(StateMachineError):
    """Exception raised when attempting to use a state machine that has not been started."""
    pass


class StateMachineAlreadyStartedError(StateMachineError):
    """Exception raised when attempting to start a state machine that has already been started."""
    pass


class PseudoStateError(StateError):
    """Exception raised for errors related to pseudostates."""
    pass


class GuardError(GotStateError):
    """Exception raised for errors related to guards."""
    pass


class ActionError(GotStateError):
    """Exception raised for errors related to actions."""
    pass


class RegionError(GotStateError):
    """Exception raised for errors related to regions."""
    pass


class PersistenceError(GotStateError):
    """Exception raised for errors related to persistence."""
    pass


class SerializationError(PersistenceError):
    """Exception raised for errors related to serialization."""
    pass


class DeserializationError(PersistenceError):
    """Exception raised for errors related to deserialization."""
    pass 