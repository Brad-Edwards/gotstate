import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from hsm.core.actions import ActionExecutionError
from hsm.core.data_management import DataManager
from hsm.core.errors import (
    ConfigurationError,
    GuardEvaluationError,
    HSMError,
    InvalidStateError,
    InvalidTransitionError,
)
from hsm.core.hooks import HookManager
from hsm.core.states import CompositeState
from hsm.core.validation import ValidationSeverity, Validator
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractStateMachine, AbstractTransition
from hsm.interfaces.types import StateID

logger = logging.getLogger(__name__)


class StateMachine(AbstractStateMachine):
    """
    Core state machine implementation.

    Runtime Invariants:
    - Only one state is active at a time
    - State transitions are atomic
    - Event processing is sequential
    - State data is isolated between states and thread-safe
    - Hooks do not affect core behavior
    - Validation occurs before start

    Attributes:
        _states: Dictionary mapping state IDs to state objects
        _transitions: List of possible transitions
        _current_state: Currently active state
        _initial_state: Starting state
        _hook_manager: Manages state/transition hooks
        _data_manager: Manages state data
        _running: Whether machine is processing events
        _logger: Logger instance
    """

    def __init__(
        self,
        states: List[AbstractState],
        transitions: List[AbstractTransition],
        initial_state: AbstractState,
    ) -> None:
        """
        Initialize the state machine.

        Args:
            states: List of all possible states
            transitions: List of allowed transitions
            initial_state: Starting state

        Raises:
            ValueError: If states or transitions are empty, if initial state is not in states list,
                       or if duplicate state IDs are found.
            TypeError: If arguments have wrong types.
        """
        if not states:
            raise ValueError("States list cannot be empty")
        if not transitions:
            raise ValueError("Transitions list cannot be empty")
        if initial_state not in states:
            raise ValueError("Initial state must be in states list")

        # Check for duplicate state IDs
        state_ids = [state.get_id() for state in states]
        duplicate_ids = {state_id for state_id in state_ids if state_ids.count(state_id) > 1}
        if duplicate_ids:
            raise ValueError(f"Duplicate state IDs found: {duplicate_ids}")

        # Initialize core components
        self._states: Dict[StateID, AbstractState] = {state.get_id(): state for state in states}
        self._transitions: List[AbstractTransition] = transitions
        self._initial_state: AbstractState = initial_state
        self._current_state: Optional[AbstractState] = None

        # Initialize managers
        self._hook_manager: HookManager = HookManager()
        self._data_manager: DataManager = DataManager()

        # Initialize state
        self._running: bool = False
        self._logger: logging.Logger = logging.getLogger("hsm.core.state_machine")

        self._initialize_state_data()
        self._validate_configuration()

    def _initialize_state_data(self) -> None:
        """Initialize the state data using the DataManager."""
        with self._data_manager.access_data() as data:
            for state_id in self._states:
                data[state_id] = {}

    def _validate_configuration(self) -> None:
        """
        Validate the state machine configuration.

        Raises:
            ConfigurationError: If validation fails
        """
        validator = Validator(list(self._states.values()), self._transitions, self._initial_state)

        # Check structure
        results = validator.validate_structure()
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR.name]
        if errors:
            raise ConfigurationError(
                "Invalid state machine structure", "structure", {r.message: r.context for r in errors}
            )

        # Check behavior
        results = validator.validate_behavior()
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR.name]
        if errors:
            raise ConfigurationError(
                "Invalid state machine behavior", "behavior", {r.message: r.context for r in errors}
            )

    def _execute_transition(self, transition: AbstractTransition, event: AbstractEvent) -> None:
        """Execute a transition."""
        source = self._current_state
        target = transition.get_target_state()

        with self._data_manager.access_data() as state_data:
            source_data = state_data[source.get_id()]
            try:
                # For self-transitions, we need special handling
                if source == target:
                    try:
                        self._hook_manager.call_on_exit(source.get_id())
                    except Exception as hook_error:
                        self._logger.warning(f"Hook error during exit (continuing): {hook_error}")
                    source.on_exit(event, state_data[source.get_id()])

                    # Execute actions
                    actions = transition.get_actions() or []
                    for action in actions:
                        action.execute(event=event, data=source_data)

                    try:
                        self._hook_manager.call_on_enter(target.get_id())
                    except Exception as hook_error:
                        self._logger.warning(f"Hook error during enter (continuing): {hook_error}")
                    target.on_entry(event, state_data[target.get_id()])

                    # For composite states, re-enter substate
                    if isinstance(target, CompositeState):
                        self._current_state = self._drill_down(target, state_data, event)
                    else:
                        self._current_state = target
                    return

                # Find common ancestor if dealing with composite states
                common_ancestor = self._find_common_ancestor(source, target)

                # Exit from current state up to (but not including) common ancestor
                current = source
                while current and current != common_ancestor:
                    try:
                        self._hook_manager.call_on_exit(current.get_id())
                    except Exception as hook_error:
                        self._logger.warning(f"Hook error during exit (continuing): {hook_error}")
                    current.on_exit(event, state_data[current.get_id()])
                    current = getattr(current, "_parent_state", None)

                # Execute transition actions
                actions = transition.get_actions() or []
                for action in actions:
                    action.execute(event=event, data=source_data)

                # Enter target state and its ancestors from common ancestor
                entry_path = []
                current = target
                while current and current != common_ancestor:
                    entry_path.insert(0, current)
                    current = getattr(current, "_parent_state", None)

                # Enter each state in the path
                for state in entry_path:
                    try:
                        self._hook_manager.call_on_enter(state.get_id())
                    except Exception as hook_error:
                        self._logger.warning(f"Hook error during enter (continuing): {hook_error}")
                    state.on_entry(event, state_data[state.get_id()])

                    # Set history only for the immediate parent of the target state
                    parent = getattr(state, "_parent_state", None)
                    if parent and isinstance(parent, CompositeState) and parent.has_history() and state == target:
                        parent.set_history_state(state)
                        parent._current_substate = state

                # If target is composite, drill down to leaf state
                if isinstance(target, CompositeState):
                    if target.has_history() and target._current_substate:
                        # Use history state if available
                        history_state = target._current_substate
                        self._current_state = history_state
                        history_state.on_entry(event, state_data[history_state.get_id()])
                    else:
                        # Otherwise drill down normally
                        self._current_state = self._drill_down(target, state_data, event)
                else:
                    self._current_state = target

            except Exception as e:
                if not isinstance(e, RuntimeError) or "Hook error" not in str(e):
                    self._handle_operation_error(
                        "Transition",
                        e,
                        {
                            "source_state": source.get_id(),
                            "target_state": target.get_id(),
                            "event": event.get_id(),
                        },
                    )

    def _find_common_ancestor(self, state1: AbstractState, state2: AbstractState) -> Optional[AbstractState]:
        """
        Find the closest common ancestor of two states, with cycle protection.
        """
        if state1 == state2:
            return state1

        # Build a set of all ancestors (and self) for state1
        ancestors1 = set()
        current = state1
        visited = {id(state1)}  # Track visited states by id to handle Mock objects

        while current is not None:
            ancestors1.add(current)
            parent = getattr(current, "_parent_state", None)
            if parent is None or id(parent) in visited:
                break
            visited.add(id(parent))
            current = parent

        # Now walk up from state2 until we find a node in ancestors1 or reach root
        current = state2
        visited = {id(state2)}  # Reset visited set for second traversal

        while current is not None:
            if current in ancestors1:
                return current
            parent = getattr(current, "_parent_state", None)
            if parent is None or id(parent) in visited:
                break
            visited.add(id(parent))
            current = parent

        return None

    def _exit_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """
        Helper function for exiting states during a transition.
        """
        current_state = self._current_state
        while current_state and current_state != source_state:
            self._hook_manager.call_on_exit(current_state.get_id())
            current_state.on_exit(event, data)
            if isinstance(current_state, CompositeState):
                current_state = current_state._parent_state if current_state._parent_state else None
            else:
                current_state = None

        if isinstance(source_state, CompositeState):
            is_target_substate = False
            if source_state.get_substates():
                for substate in source_state.get_substates():
                    if substate == target_state:
                        is_target_substate = True
                        break

            # Exit current substate if it exists
            if source_state._current_substate:
                self._hook_manager.call_on_exit(source_state._current_substate.get_id())
                source_state._current_substate.on_exit(event, data)
                if not is_target_substate:
                    source_state._current_substate = None

            # Only exit the composite parent if the target is *not* a substate
            if not is_target_substate:
                self._hook_manager.call_on_exit(source_state.get_id())
                source_state.on_exit(event, data)
                self._current_state = None

    def _enter_states(
        self, source_state: AbstractState, target_state: AbstractState, event: AbstractEvent, data: Any
    ) -> None:
        """Enter states from common ancestor to target."""
        entry_path = []
        current = target_state
        while current and current != source_state:
            entry_path.insert(0, current)
            current = getattr(current, "_parent_state", None)

        for state in entry_path:
            self._hook_manager.call_on_enter(state.get_id())
            state.on_entry(event, data)
            if isinstance(state, CompositeState):
                state._enter_substate(event, data)
                if state.has_history():
                    state.set_history_state(state._current_substate)

        self._current_state = target_state

    def _find_valid_transition(self, event: AbstractEvent) -> Optional[AbstractTransition]:
        """Find the highest priority valid transition for the current event."""
        if not self._current_state:
            return None

        valid_transitions = []

        for transition in self._transitions:
            source_state = transition.get_source_state()
            if source_state == self._current_state or (
                isinstance(self._current_state, CompositeState)
                and source_state == self._current_state._current_substate
            ):
                guard = transition.get_guard()
                with self._data_manager.access_data() as state_data:  # Use DataManager for thread-safety
                    if guard:
                        try:
                            source_data = state_data[source_state.get_id()]
                            if guard.check(event, source_data):
                                valid_transitions.append(transition)
                        except Exception as e:
                            logger.error(
                                "Guard check failed for transition from '%s' to '%s' in state '%s': %s",
                                source_state.get_id(),
                                transition.get_target_state().get_id(),
                                self._current_state.get_id(),
                                str(e),
                            )
                            continue
                    else:
                        valid_transitions.append(transition)

        if not valid_transitions:
            return None

        return max(valid_transitions, key=lambda t: t.get_priority())

    def get_state(self) -> Optional[AbstractState]:
        """
        Get the current state.

        Returns:
            Current state object or None if no current state

        """
        return self._current_state

    def start(self) -> None:
        """Start the state machine."""
        if self._running:
            raise InvalidStateError(
                "State machine already running", self.get_state().get_id() if self.get_state() else None, "start"
            )

        self._running = True
        self._current_state = self._initial_state
        self._initialize_state_data()

        with self._data_manager.access_data() as state_data:
            try:
                # Enter initial state
                try:
                    self._hook_manager.call_on_enter(self._initial_state.get_id())
                except Exception as hook_error:
                    self._logger.warning(f"Hook error during start (continuing): {hook_error}")

                # Enter the initial state
                self._initial_state.on_entry(None, state_data[self._initial_state.get_id()])

                # If initial state is composite, handle history and drill down
                if isinstance(self._initial_state, CompositeState):
                    # Enter initial substate if needed
                    if not self._initial_state._current_substate:
                        self._initial_state._enter_substate(None, state_data[self._initial_state.get_id()])

                    # Now drill down to leaf state
                    self._current_state = self._drill_down(
                        self._initial_state, state_data, None
                    )  # Pass None as the event
                else:
                    self._current_state = self._initial_state

            except Exception as e:
                self._running = False
                raise InvalidStateError(
                    f"Failed to enter initial state: {str(e)}", self._initial_state.get_id(), "enter", {"error": str(e)}
                )

    def stop(self) -> None:
        """Stop the state machine."""
        if not self._running:
            raise InvalidStateError("State machine not running", None, "stop")

        if self._current_state:
            with self._data_manager.access_data() as state_data:
                self._exit_up(self._current_state, state_data)

        self._running = False
        self._current_state = None

    def process_event(self, event: AbstractEvent) -> None:
        """Process an event, potentially triggering a transition."""
        if not self._running:
            raise InvalidStateError("Cannot process events while stopped", None, "process_event")

        if not self._current_state:
            raise InvalidStateError("No current state", None, "process_event")

        # Find eligible transitions
        transition = self._find_valid_transition(event)

        if transition:
            # Execute transition
            self._execute_transition(transition, event)

    def get_current_state_id(self) -> StateID:
        """
        Get the ID of the current state.

        Returns:
            Current state ID or 'None' if no current state

        Raises:
            InvalidStateError: If no current state
        """
        if not self._current_state:
            raise InvalidStateError("No current state", "None", "get_current_state_id")
        return self._current_state.get_id()

    def _handle_operation_error(self, operation: str, err: Exception, details: Optional[dict] = None) -> None:
        """
        Handle an operation error by logging it and optionally raising it.
        """
        # Adjusted to match the placeholder-based test expectation
        self._logger.error("Error during operation '%s': %s", operation, err, exc_info=True)

        if details is not None:
            self._logger.error("Error details: %s", details)

        # Existing error-raising logic remains unchanged below
        raise err

    def _drill_down(
        self, top_state: AbstractState, data: Dict[str, Any], event: Optional[AbstractEvent]
    ) -> AbstractState:
        """
        Recursively enter initial composite substates until reaching a leaf state.
        Updates the current state to the deepest active leaf state.

        Args:
            top_state: The state to start drilling from
            data: The state data dictionary
            event: The event to pass to the on_entry functions

        Returns:
            The deepest leaf state that was entered
        """
        current = top_state
        while isinstance(current, CompositeState):
            # Get the current substate
            next_state = current._current_substate

            # If no substate, we're done
            if not next_state:
                break

            # Enter the substate if not already entered
            if self._current_state != next_state:
                try:
                    self._hook_manager.call_on_enter(next_state.get_id())
                except Exception as hook_error:
                    self._logger.warning(f"Hook error during substate enter (continuing): {hook_error}")

                next_state.on_entry(event, data[next_state.get_id()])

                # Set history for this level if supported
                if current.has_history():
                    current.set_history_state(next_state)

            current = next_state
            self._current_state = current  # Update current state as we drill down

        return current

    def _exit_up(self, leaf_state: AbstractState, data: Dict[str, Any]) -> None:
        """
        Recursively exit from a leaf state up through its parent(s).
        """
        current = leaf_state
        while current is not None:
            try:
                self._hook_manager.call_on_exit(current.get_id())
            except Exception as hook_error:
                self._logger.warning(f"Hook error during exit (continuing): {hook_error}")
            current.on_exit(None, data[current.get_id()])
            # Move one level up
            current = getattr(current, "_parent_state", None)
