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
    """Base exception for executor-related errors."""
    pass


class StateError(HSMError):
    """Base class for state-related errors."""
    
    def __init__(
        self, 
        message: str, 
        state_info: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.__dict__.update(state_info)


class InvalidTransitionError(StateError):
    """Raised when a state transition is invalid."""
    
    def __init__(
        self, 
        message: str, 
        source_state: str, 
        target_state: str, 
        event: Any,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        state_info = {
            "source_state": source_state,
            "target_state": target_state,
            "event": event
        }
        super().__init__(message, state_info, details)


class InvalidStateError(StateError):
    """Raised when operations are performed on an invalid state."""
    
    def __init__(
        self, 
        message: str, 
        state_id: str, 
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        state_info = {
            "state_id": state_id,
            "operation": operation
        }
        super().__init__(message, state_info, details)


class ValidationError(HSMError):
    """Base class for validation-related errors."""
    
    def __init__(
        self,
        message: str,
        validation_info: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.__dict__.update(validation_info)


class ConfigurationError(ValidationError):
    """Raised when state machine configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        component: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        validation_info = {
            "component": component,
            "validation_errors": validation_errors or {}
        }
        super().__init__(message, validation_info, details)


class ExecutionError(HSMError):
    """Base class for execution-related errors."""
    
    def __init__(
        self,
        message: str,
        execution_info: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.__dict__.update(execution_info)


class GuardEvaluationError(ExecutionError):
    """Raised when a guard condition evaluation fails."""
    
    def __init__(
        self,
        message: str,
        guard_name: str,
        state_data: Any,
        event: Any,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        execution_info = {
            "guard_name": guard_name,
            "state_data": state_data,
            "event": event
        }
        super().__init__(message, execution_info, details)


class ActionExecutionError(ExecutionError):
    """Raised when a transition action fails to execute."""
    
    def __init__(
        self,
        message: str,
        action_name: str,
        state_data: Any,
        event: Any,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        execution_info = {
            "action_name": action_name,
            "state_data": state_data,
            "event": event
        }
        super().__init__(message, execution_info, details)


class ConcurrencyError(HSMError):
    """Raised when concurrent operations conflict."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.operation = operation
        self.resource = resource


class EventQueueFullError(HSMError):
    """Raised when event queue capacity is exceeded."""
    
    def __init__(
        self,
        message: str,
        queue_size: int,
        max_size: int,
        dropped_event: Any,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self.queue_size = queue_size
        self.max_size = max_size
        self.dropped_event = dropped_event


@dataclass(frozen=True)
class ErrorContext:
    """Container for error context information."""
    error_type: Type[HSMError]
    timestamp: float
    traceback: str
    details: Dict[str, Any]


def create_error_context(error: HSMError, traceback: str) -> ErrorContext:
    """Creates an error context object for logging and debugging."""
    import time
    return ErrorContext(
        error_type=type(error),
        timestamp=time.time(),
        traceback=traceback,
        details=error.details
    )
