class HSMError(Exception):
    pass


class InvalidTransitionError(HSMError):
    pass


class InvalidStateError(HSMError):
    pass


class ConfigurationError(HSMError):
    pass


class GuardEvaluationError(HSMError):
    pass


class ActionExecutionError(HSMError):
    pass


class ConcurrencyError(HSMError):
    pass


class EventQueueFullError(HSMError):
    pass
