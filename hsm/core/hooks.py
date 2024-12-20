class HookManager:
    """
    Manages the registration and execution of hooks that listen to state machine
    lifecycle events (on_enter, on_exit, on_error). Users can attach logging,
    monitoring, or custom side effects without altering core logic.
    """

    def __init__(self, hooks: list["HookProtocol"] = None) -> None:
        """
        Initialize with an optional list of hook objects.
        """
        raise NotImplementedError()

    def register_hook(self, hook: "HookProtocol") -> None:
        """
        Add a new hook to the manager's list of hooks.

        :param hook: An object implementing HookProtocol methods.
        """
        raise NotImplementedError()

    def execute_on_enter(self, state: "State") -> None:
        """
        Run all hooks' on_enter logic when entering a state.
        """
        raise NotImplementedError()

    def execute_on_exit(self, state: "State") -> None:
        """
        Run all hooks' on_exit logic when exiting a state.
        """
        raise NotImplementedError()

    def execute_on_error(self, error: Exception) -> None:
        """
        Run all hooks' on_error logic when an exception occurs.
        """
        raise NotImplementedError()


class _HookInvoker:
    """
    Internal helper that iterates through a list of hooks and invokes their
    lifecycle methods in a controlled manner.
    """

    def __init__(self, hooks: list["HookProtocol"]) -> None:
        """
        Store hooks for invocation.
        """
        raise NotImplementedError()

    def invoke_on_enter(self, state: "State") -> None:
        """
        Call each hook's on_enter method.
        """
        raise NotImplementedError()

    def invoke_on_exit(self, state: "State") -> None:
        """
        Call each hook's on_exit method.
        """
        raise NotImplementedError()

    def invoke_on_error(self, error: Exception) -> None:
        """
        Call each hook's on_error method.
        """
        raise NotImplementedError()
