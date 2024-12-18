# hsm/core/errors.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type


def _create_info_dict(**kwargs) -> Dict[str, Any]:
    """Create an info dictionary from keyword arguments, filtering out None values."""
    return {k: v for k, v in kwargs.items() if v is not None}


class HSMError(Exception):
    """Base exception class for all HSM-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def _set_attributes(self, **kwargs) -> None:
        """Set instance attributes from keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)


class ExecutorError(HSMError):
    """Base exception for executor-related errors."""

    pass


class StateError(HSMError):
    """Base class for state-related errors."""

    def __init__(self, message: str, state_info: Dict[str, Any], details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details)
        self._set_attributes(**state_info)


class InvalidTransitionError(StateError):
    """Raised when a state transition is invalid."""

    def __init__(
        self, message: str, source_state: str, target_state: str, event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        state_info = _create_info_dict(source_state=source_state, target_state=target_state, event=event)
        super().__init__(message, state_info, details)


class InvalidStateError(StateError):
    """Raised when operations are performed on an invalid state."""

    def __init__(self, message: str, state_id: str, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        state_info = _create_info_dict(state_id=state_id, operation=operation)
        super().__init__(message, state_info, details)


class ValidationError(HSMError):
    """Raised when validation fails."""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        validation_results: Optional[List[Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self._set_attributes(component=component, validation_results=validation_results or [])


class ConfigurationError(ValidationError):
    """Raised when state machine configuration is invalid."""

    def __init__(
        self,
        message: str,
        component: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, component=component, details=details)
        self.validation_errors = validation_errors or {}


class ExecutionError(HSMError):
    """Base class for execution-related errors."""

    def __init__(self, message: str, execution_info: Dict[str, Any], details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details)
        self._set_attributes(**execution_info)


class GuardEvaluationError(ExecutionError):
    """Raised when a guard condition evaluation fails."""

    def __init__(
        self, message: str, guard_name: str, state_data: Any, event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        execution_info = _create_info_dict(guard_name=guard_name, state_data=state_data, event=event)
        super().__init__(message, execution_info, details)


class ActionExecutionError(ExecutionError):
    """Raised when a transition action fails to execute."""

    def __init__(
        self, message: str, action_name: str, state_data: Any, event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        execution_info = _create_info_dict(action_name=action_name, state_data=state_data, event=event)
        super().__init__(message, execution_info, details)


class ConcurrencyError(HSMError):
    """Raised when concurrent operations conflict."""

    def __init__(self, message: str, operation: str, resource: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, details)
        self._set_attributes(operation=operation, resource=resource)


class EventQueueFullError(HSMError):
    """Raised when event queue capacity is exceeded."""

    def __init__(
        self, message: str, queue_size: int, max_size: int, dropped_event: Any, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details)
        self._set_attributes(queue_size=queue_size, max_size=max_size, dropped_event=dropped_event)


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

    return ErrorContext(error_type=type(error), timestamp=time.time(), traceback=traceback, details=error.details)
