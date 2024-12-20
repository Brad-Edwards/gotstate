# hsm/core/errors.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
class HSMError(Exception):
    """
    Base exception class for errors within the hierarchical state machine library.
    """

    raise NotImplementedError()


class StateNotFoundError(HSMError):
    """
    Raised when a requested state does not exist in the machine or hierarchy.
    """

    raise NotImplementedError()


class TransitionError(HSMError):
    """
    Raised when an attempted state transition is invalid or cannot be completed.
    """

    raise NotImplementedError()


class ValidationError(HSMError):
    """
    Raised when validation detects configuration or runtime constraints violations.
    """

    raise NotImplementedError()
