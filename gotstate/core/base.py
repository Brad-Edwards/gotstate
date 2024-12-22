from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(eq=False)
class StateBase:
    """
    Base class for hierarchical state machine states.

    This class provides the fundamental structure and behavior for states within
    a hierarchical state machine. Each state can have a parent state, entry actions
    that execute when the state is entered, and exit actions that execute when the
    state is exited.

    Attributes:
        name: A unique identifier for the state.
        parent: Reference to the parent state in the hierarchy, or None if this is a root state.
        entry_actions: List of callables that execute when entering this state.
        exit_actions: List of callables that execute when exiting this state.

    Example:
        ```python
        def on_enter_idle():
            print("Entering idle state")

        def on_exit_idle():
            print("Exiting idle state")

        idle_state = StateBase(
            name="idle",
            entry_actions=[on_enter_idle],
            exit_actions=[on_exit_idle]
        )
        ```
    """

    name: str
    parent: Optional["StateBase"] = None
    entry_actions: List[callable] = field(default_factory=list)
    exit_actions: List[callable] = field(default_factory=list)

    def __hash__(self) -> int:
        """
        Generate a unique hash for the state based on its name and memory address.

        The hash is used for state comparison and container operations. It ensures
        that states with the same name but different positions in the hierarchy
        are treated as distinct.

        Returns:
            int: A unique hash value for this state.
        """
        # Use object id to break cycles while maintaining uniqueness
        return hash((self.name, id(self)))

    def __eq__(self, other: object) -> bool:
        """
        Compare this state with another for equality.

        States are considered equal only if they are the exact same object instance.
        This prevents ambiguity when the same state name appears in different
        parts of the state hierarchy.

        Args:
            other: The object to compare with this state.

        Returns:
            bool: True if the states are the same instance, False otherwise.
        """
        if not isinstance(other, StateBase):
            return NotImplemented
        return id(self) == id(other)

    def on_enter(self) -> None:
        """
        Execute all entry actions for this state.

        This method is called automatically by the state machine when transitioning
        into this state. It executes all registered entry actions in the order
        they were added.

        Note:
            Entry actions should be quick operations to avoid blocking the state
            machine. For long-running operations, consider using events or
            background tasks.
        """
        for action in self.entry_actions:
            action()

    def on_exit(self) -> None:
        """
        Execute all exit actions for this state.

        This method is called automatically by the state machine when transitioning
        out of this state. It executes all registered exit actions in the order
        they were added.

        Note:
            Exit actions should be quick operations to avoid blocking the state
            machine. For long-running operations, consider using events or
            background tasks.
        """
        for action in self.exit_actions:
            action()
