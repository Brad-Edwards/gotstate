"""
Exception hierarchy for gotstate.

Provides structured, typed exceptions for all error conditions in the
hierarchical state machine library.
"""


class GotStateError(Exception):
    """Base exception for all gotstate errors."""


class StateError(GotStateError):
    """Errors related to state operations."""


class InvalidStateError(StateError):
    """A state is in an invalid configuration."""


class StateNotFoundError(StateError):
    """A referenced state does not exist."""


class DuplicateStateError(StateError):
    """A state with the same name already exists in scope."""


class TransitionError(GotStateError):
    """Errors related to transitions."""


class InvalidTransitionError(TransitionError):
    """A transition has an invalid configuration."""


class GuardError(TransitionError):
    """A guard condition raised an error during evaluation."""


class EventError(GotStateError):
    """Errors related to event processing."""


class InvalidEventError(EventError):
    """An event has an invalid configuration."""


class EventQueueFullError(EventError):
    """The event queue has reached its capacity."""


class RegionError(GotStateError):
    """Errors related to region operations."""


class MachineError(GotStateError):
    """Errors related to state machine operations."""


class MachineNotInitializedError(MachineError):
    """The state machine has not been initialized."""


class MachineAlreadyRunningError(MachineError):
    """The state machine is already running."""


class ValidationError(GotStateError):
    """Errors related to validation."""


class SerializationError(GotStateError):
    """Errors related to serialization/deserialization."""
