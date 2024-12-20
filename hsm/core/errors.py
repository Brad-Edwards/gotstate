### hsm/core/errors.py
class HSMError(Exception):
    """
    Base exception class for errors within the hierarchical state machine library.
    """


class StateNotFoundError(HSMError):
    """
    Raised when a requested state does not exist in the machine or hierarchy.
    """


class TransitionError(HSMError):
    """
    Raised when an attempted state transition is invalid or cannot be completed.
    """


class ValidationError(HSMError):
    """
    Raised when validation detects configuration or runtime constraints violations.
    """
