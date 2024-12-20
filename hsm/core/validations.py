### hsm/core/validation.py
class Validator:
    """
    Performs construction-time and runtime validation of the state machine,
    ensuring states, transitions, and events conform to defined rules.
    """

    def __init__(self) -> None:
        """
        Initialize the validator, potentially loading default or custom rules.
        """

    def validate_state_machine(self, machine: "StateMachine") -> None:
        """
        Check the machine's states and transitions for consistency.

        :param machine: The state machine to validate.
        """

    def validate_transition(self, transition: "Transition") -> None:
        """
        Check that a given transition is well-formed.

        :param transition: The transition to validate.
        """

    def validate_event(self, event: "Event") -> None:
        """
        Validate that an event is well-defined and usable.

        :param event: The event to validate.
        """


class _ValidationRulesEngine:
    """
    Internal engine applying a set of validation rules to states, transitions,
    and events. Centralizes validation logic for easier maintenance.
    """

    def __init__(self) -> None:
        """
        Initialize internal rule sets.
        """

    def validate_machine(self, machine: "StateMachine") -> None:
        """
        Apply all machine-level validation rules.
        """

    def validate_transition(self, transition: "Transition") -> None:
        """
        Apply transition-level validation rules.
        """

    def validate_event(self, event: "Event") -> None:
        """
        Apply event-level validation rules.
        """


class _DefaultValidationRules:
    """
    Provides built-in validation rules ensuring basic correctness of states,
    transitions, and events out of the box.
    """

    @staticmethod
    def validate_machine(machine: "StateMachine") -> None:
        """
        Check for unreachable states, invalid references, etc.
        """
        raise NotImplementedError()

    @staticmethod
    def validate_transition(transition: "Transition") -> None:
        """
        Check that transition source/target states exist and guards are callable.
        """
        raise NotImplementedError()

    @staticmethod
    def validate_event(event: "Event") -> None:
        """
        Check that event names are non-empty and metadata is valid.
        """
        raise NotImplementedError()
