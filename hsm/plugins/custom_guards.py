# Example plugin file: hsm/plugins/custom_guards.py
class MyCustomGuard:
    """
    A user-defined guard example that checks a custom condition before allowing a transition.
    """

    def __init__(self, condition_fn: callable) -> None:
        """
        Initialize with a condition function returning bool.
        """
        raise NotImplementedError()

    def check(self, event: "Event") -> bool:
        """
        Evaluate the custom condition with the given event.

        :param event: The triggering event.
        :return: True if allowed, False if not.
        """
        raise NotImplementedError()
