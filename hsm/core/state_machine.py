# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from hsm.core.events import Event
from hsm.core.hooks import HookProtocol
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.core.validations import Validator


class StateMachine:
    """
    A synchronous hierarchical state machine that manages state transitions
    based on incoming events. It starts in an initial state and processes events
    using defined transitions, optionally validated and hooked into by plugins.

    The StateMachine ensures that only valid state transitions occur and that
    entry/exit actions and hooks are consistently executed.
    """

    def __init__(
        self, initial_state: "State", validator: "Validator" = None, hooks: list["HookProtocol"] = None
    ) -> None:
        """
        Initialize the state machine with an initial state, optional validator,
        and hooks for monitoring or logging.

        :param initial_state: The starting state of the machine.
        :param validator: Optional Validator instance for construction/runtime checks.
        :param hooks: Optional list of hooks to execute on enter/exit/error events.
        """
        raise NotImplementedError()

    @property
    def current_state(self) -> "State":
        """
        Return the current active state of the state machine.

        :return: The current State instance.
        """
        raise NotImplementedError()

    def start(self) -> None:
        """
        Start the state machine, executing initial entry actions and performing
        any configured validations before beginning to process events.
        """
        raise NotImplementedError()

    def process_event(self, event: "Event") -> None:
        """
        Process an incoming event, performing state transitions if applicable.

        :param event: An Event instance triggering potential transitions.
        """
        raise NotImplementedError()

    def stop(self) -> None:
        """
        Gracefully stop the state machine, executing exit actions and hooks.
        This may be used to clean up resources and prepare for shutdown.
        """
        raise NotImplementedError()


class CompositeStateMachine(StateMachine):
    """
    A hierarchical extension of StateMachine that can contain nested submachines
    under composite states. This allows complex modeling of state hierarchies,
    enabling finer-grained control and more manageable state definitions.
    """

    def add_submachine(self, state: "CompositeState", submachine: "StateMachine") -> None:
        """
        Associate a subordinate StateMachine with a CompositeState, allowing
        nested behaviors within that state.

        :param state: The composite state serving as a container.
        :param submachine: The sub-state machine controlling nested states.
        """
        raise NotImplementedError()

    @property
    def submachines(self) -> dict["CompositeState", "StateMachine"]:
        """
        Retrieve a mapping of CompositeStates to their associated submachines.

        :return: Dictionary mapping composite states to their nested state machines.
        """
        raise NotImplementedError()


class _StateMachineContext:
    """
    Internal context management for StateMachine, tracking current state and
    available transitions. Not for direct use by library clients.
    """

    def __init__(self) -> None:
        """
        Initialize internal structures to manage the current state and transitions.
        """
        raise NotImplementedError()

    def get_current_state(self) -> "State":
        """
        Internal getter for current state.
        """
        raise NotImplementedError()

    def set_current_state(self, state: "State") -> None:
        """
        Internal setter to update the current state reference.
        """
        raise NotImplementedError()

    def get_transitions(self) -> list["Transition"]:
        """
        Retrieve all transitions known to this state machine context.
        """
        raise NotImplementedError()

    def add_transition(self, transition: "Transition") -> None:
        """
        Add a transition to the internal transition list.
        """
        raise NotImplementedError()


class _ErrorRecoveryStrategy:
    """
    Abstract interface for custom error recovery strategies. Provides a hook
    to handle exceptions within the state machine lifecycle.
    """

    def recover(self, error: Exception, state_machine: StateMachine) -> None:
        """
        Attempt to recover the state machine from an error by adjusting state
        or performing fallback actions.

        :param error: The exception raised.
        :param state_machine: The state machine instance encountering the error.
        """
        raise NotImplementedError()
