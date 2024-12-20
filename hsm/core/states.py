### hsm/core/states.py
class State:
    """
    Represents a named state within the machine. It may define entry and exit
    actions, as well as hold its own data dictionary for state-specific variables.
    """

    def __init__(
        self, name: str, entry_actions: list["ActionProtocol"] = None, exit_actions: list["ActionProtocol"] = None
    ) -> None:
        """
        Initialize the state with a name and optional lists of actions to run on
        entry and exit.

        :param name: Unique name identifying this state.
        :param entry_actions: Actions executed upon entering this state.
        :param exit_actions: Actions executed upon exiting this state.
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """
        Return the state's name.
        """
        raise NotImplementedError()

    @property
    def data(self) -> dict[str, any]:
        """
        Access the state's local data store, which may be used to keep stateful
        information relevant only while this state is active.
        """
        raise NotImplementedError()

    def on_enter(self) -> None:
        """
        Called when the state machine transitions into this state. Executes any
        defined entry actions and initializes state data.
        """
        raise NotImplementedError()

    def on_exit(self) -> None:
        """
        Called when leaving this state. Executes exit actions and cleans up
        any state data if needed.
        """
        raise NotImplementedError()

    def initialize_data(self) -> None:
        """
        Prepare the state's data dictionary when this state is entered.
        """
        raise NotImplementedError()

    def cleanup_data(self) -> None:
        """
        Clean up or reset the state's data dictionary when exiting this state.
        """
        raise NotImplementedError()


class CompositeState(State):
    """
    A state containing child states, representing a hierarchical structure.
    Useful for grouping related states and transitions within a logical namespace.
    """

    def add_child_state(self, state: "State") -> None:
        """
        Add a child state to the composite state. Child states can form a nested
        hierarchy, enabling complex, modular state machines.

        :param state: The state to add as a child.
        """
        raise NotImplementedError()

    def get_child_state(self, name: str) -> "State":
        """
        Retrieve a child state by name.

        :param name: Name of the desired child state.
        :return: The child State instance.
        """
        raise NotImplementedError()

    @property
    def children(self) -> dict[str, "State"]:
        """
        Obtain a dictionary of child states keyed by their names.
        """
        raise NotImplementedError()


class _StateDataWrapper:
    """
    Internal helper class to manage and lock access to state data. Ensures that
    updates to data are consistent and thread-safe.
    """

    def __init__(self, state: State) -> None:
        """
        Initialize with a reference to the parent State instance.
        """
        raise NotImplementedError()

    def get_data(self) -> dict[str, any]:
        """
        Retrieve the state's internal data dictionary, potentially under a lock.
        """
        raise NotImplementedError()

    def set_data(self, data: dict[str, any]) -> None:
        """
        Set the state's data dictionary, used internally for atomic updates.
        """
        raise NotImplementedError()


class _EntryExitActionExecutor:
    """
    Internal utility that cleanly executes a list of actions when entering or
    exiting a state. Handles errors gracefully and ensures actions run in order.
    """

    def __init__(self, entry_actions: list["ActionProtocol"], exit_actions: list["ActionProtocol"]) -> None:
        """
        Store the lists of actions to be executed on entering and exiting a state.
        """
        raise NotImplementedError()

    def execute_entry_actions(self) -> None:
        """
        Run all entry actions sequentially.
        """
        raise NotImplementedError()

    def execute_exit_actions(self) -> None:
        """
        Run all exit actions sequentially.
        """
        raise NotImplementedError()
