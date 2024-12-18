# hsm/core/errors.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type


class HSMError(Exception):
    """Base exception class for all HSM-related errors.

    Attributes:
        message: Error description
        details: Additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ExecutorError(HSMError):
    """Base exception for executor-related errors.

    Attributes:
        message: Error description
        details: Additional error context
    """

    pass


class InvalidTransitionError(HSMError):
    """Raised when a state transition is invalid.

    Attributes:
        source_state: ID of source state
        target_state: ID of target state
        event: Event that triggered the transition
    """

    def __init__(
        self, message: str, source_state: str, target_state: str, event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.source_state = source_state
        self.target_state = target_state
        self.event = event


class InvalidStateError(HSMError):
    """Raised when operations are performed on an invalid state.

    Attributes:
        state_id: ID of the invalid state
        operation: Operation that failed
    """

    def __init__(self, message: str, state_id: str, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details)
        self.state_id = state_id
        self.operation = operation


class ConfigurationError(HSMError):
    """Raised when state machine configuration is invalid.

    Attributes:
        component: Component with invalid configuration
        validation_errors: List of validation failures
    """

    def __init__(
        self,
        message: str,
        component: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.component = component
        self.validation_errors = validation_errors or {}


class GuardEvaluationError(HSMError):
    """Raised when a guard condition evaluation fails.

    Attributes:
        guard_name: Name of the failed guard
        state_data: State data during evaluation
        event: Event being processed
    """

    def __init__(
        self, message: str, guard_name: str, state_data: Any, event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.guard_name = guard_name
        self.state_data = state_data
        self.event = event


class ActionExecutionError(HSMError):
    """Raised when a transition action fails to execute.

    Attributes:
        action_name: Name of the failed action
        state_data: State data during execution
        event: Event being processed
    """

    def __init__(
        self, message: str, action_name: str, state_data: Any, event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.action_name = action_name
        self.state_data = state_data
        self.event = event


class ConcurrencyError(HSMError):
    """Raised when concurrent operations conflict.

    Attributes:
        operation: Operation that failed
        resource: Resource that caused the conflict
    """

    def __init__(self, message: str, operation: str, resource: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details)
        self.operation = operation
        self.resource = resource


class EventQueueFullError(HSMError):
    """Raised when event queue capacity is exceeded.

    Attributes:
        queue_size: Current size of the queue
        max_size: Maximum allowed size
        dropped_event: Event that couldn't be queued
    """

    def __init__(
        self, message: str, queue_size: int, max_size: int, dropped_event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.queue_size = queue_size
        self.max_size = max_size
        self.dropped_event = dropped_event


class ValidationError(HSMError):
    """Raised when validation fails.

    Attributes:
        component: Component that failed validation
        validation_results: List of validation failures
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        validation_results: Optional[List[Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.component = component
        self.validation_results = validation_results or []


@dataclass(frozen=True)
class ErrorContext:
    """Container for error context information."""

    error_type: Type[HSMError]
    timestamp: float
    traceback: str
    details: Dict[str, Any]


def create_error_context(error: HSMError, traceback: str) -> ErrorContext:
    """Creates an error context object for logging and debugging.

    Args:
        error: The HSM error that occurred
        traceback: String representation of the traceback

    Returns:
        ErrorContext object containing error details
    """
    import time

    return ErrorContext(error_type=type(error), timestamp=time.time(), traceback=traceback, details=error.details)
