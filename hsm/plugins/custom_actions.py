# Example plugin file: hsm/plugins/custom_actions.py
class MyCustomAction:
    """
    A user-defined action example that executes a custom piece of code during a transition.
    """

    def __init__(self, action_fn: callable) -> None:
        """
        Initialize with a custom action function.
        """
        raise NotImplementedError()

    def run(self, event: "Event") -> None:
        """
        Perform the action in response to the given event.

        :param event: The triggering event.
        """
        raise NotImplementedError()
