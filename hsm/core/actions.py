### hsm/core/actions.py
class BasicActions:
    """
    Provides simple, built-in action handlers. Users can extend or create custom
    actions in plugins.
    """

    @staticmethod
    def execute(action_fn: callable, **kwargs) -> None:
        """
        Execute an action function with optional keyword arguments.

        :param action_fn: A callable representing the action's logic.
        :param kwargs: Additional parameters for the action.
        """
        raise NotImplementedError()


class _ActionAdapter:
    """
    Internal adapter that wraps a user-defined callable into an ActionProtocol,
    ensuring consistent action invocation.
    """

    def __init__(self, action_fn: callable) -> None:
        """
        Wrap an action function for consistent execution.
        """

    def run(self, event: "Event") -> None:
        """
        Execute the action for the given event.
        """
        raise NotImplementedError()
