from typing import Any, Dict, Generator
from unittest.mock import Mock, call, patch

import pytest
from pytest import LogCaptureFixture

from hsm.core.errors import ConfigurationError, InvalidStateError
from hsm.core.state_machine import StateMachine
from hsm.core.states import CompositeState
from hsm.core.validation import ValidationSeverity
from hsm.interfaces.abc import AbstractEvent, AbstractState, AbstractTransition
from hsm.core.events import Event


@pytest.fixture
def mock_state() -> Any:
    s = Mock(spec=AbstractState)
    s.get_id.return_value = "state1"
    s.on_entry.return_value = None
    s.on_exit.return_value = None
    return s


@pytest.fixture
def mock_initial_state() -> Any:
    s = Mock(spec=AbstractState)
    s.get_id.return_value = "initial_state"
    s.on_entry.return_value = None
    s.on_exit.return_value = None
    return s


@pytest.fixture
def mock_other_state() -> Any:
    s = Mock(spec=AbstractState)
    s.get_id.return_value = "other_state"
    s.on_entry.return_value = None
    s.on_exit.return_value = None
    return s


@pytest.fixture
def mock_event() -> Any:
    e = Mock(spec=AbstractEvent)
    e.get_id.return_value = "event1"
    return e


@pytest.fixture
def mock_transition(mock_initial_state: Any, mock_other_state: Any) -> Any:
    t = Mock(spec=AbstractTransition)
    t.get_source_state.return_value = mock_initial_state
    t.get_target_state.return_value = mock_other_state
    t.get_guard.return_value = None
    t.get_actions.return_value = []
    t.get_priority.return_value = 1
    return t


@pytest.fixture
def valid_sm(mock_initial_state: Any, mock_other_state: Any, mock_transition: Any) -> StateMachine:
    states = [mock_initial_state, mock_other_state]
    transitions = [mock_transition]
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []
        sm = StateMachine(states, transitions, mock_initial_state)
    return sm


@pytest.fixture
def test_sm(valid_sm: StateMachine) -> StateMachine:
    """Fixture for testing internal methods"""
    return valid_sm


def test_initialization_success(valid_sm, mock_initial_state):
    assert valid_sm.get_state() is None
    # After construction, not started yet.
    assert valid_sm._running is False
    assert valid_sm._initial_state == mock_initial_state


def test_empty_states():
    with pytest.raises(ValueError, match="States list cannot be empty"):
        StateMachine([], [Mock(spec=AbstractTransition)], Mock(spec=AbstractState))


def test_empty_transitions(mock_state):
    with pytest.raises(ValueError, match="Transitions list cannot be empty"):
        StateMachine([mock_state], [], mock_state)


def test_initial_state_not_in_list(mock_state):
    s1 = mock_state
    s1.get_id.return_value = "s1"
    s2 = Mock(spec=AbstractState)
    s2.get_id.return_value = "s2"
    with pytest.raises(ValueError, match="Initial state must be in states list"):
        StateMachine([s1], [Mock(spec=AbstractTransition)], s2)


def test_validation_error(mock_initial_state):
    s = mock_initial_state
    s.get_id.return_value = "init"
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = [
            Mock(severity=ValidationSeverity.ERROR.name, message="Structure error", context={})
        ]
        with pytest.raises(ConfigurationError, match="Invalid state machine structure"):
            StateMachine([s], [Mock(spec=AbstractTransition)], s)


def test_start_stop_cycle(valid_sm, mock_initial_state):
    valid_sm.start()
    assert valid_sm._running is True
    assert valid_sm.get_state() == mock_initial_state
    mock_initial_state.on_entry.assert_called_once()

    valid_sm.stop()
    assert valid_sm._running is False
    assert valid_sm.get_state() is None
    # stop should call on_exit via _exit_states


def test_start_twice(valid_sm):
    valid_sm.start()
    with pytest.raises(InvalidStateError, match="State machine already running"):
        valid_sm.start()


def test_stop_without_start(valid_sm):
    with pytest.raises(InvalidStateError, match="State machine not running"):
        valid_sm.stop()


def test_get_current_state_id_no_state(valid_sm):
    with pytest.raises(InvalidStateError, match="No current state"):
        valid_sm.get_current_state_id()


def test_get_current_state_id_after_start(valid_sm):
    valid_sm.start()
    assert valid_sm.get_current_state_id() == "initial_state"


def test_process_event_when_stopped(valid_sm, mock_event):
    with pytest.raises(InvalidStateError, match="Cannot process events while stopped"):
        valid_sm.process_event(mock_event)


def test_process_event_no_current_state(valid_sm, mock_event):
    # Start and then artificially set current state to None
    valid_sm.start()
    valid_sm._current_state = None
    with pytest.raises(InvalidStateError, match="No current state"):
        valid_sm.process_event(mock_event)


def test_valid_transition_trigger(valid_sm, mock_event, mock_initial_state, mock_transition, mock_other_state):
    # Configure the transition to be triggered by our event
    mock_transition.get_source_state.return_value = mock_initial_state
    mock_transition.get_target_state.return_value = mock_other_state
    mock_transition.get_guard.return_value = None
    mock_transition.get_actions.return_value = []

    valid_sm.start()
    assert valid_sm.get_current_state_id() == "initial_state"

    # Process the event
    valid_sm.process_event(mock_event)

    # Verify the transition occurred
    mock_initial_state.on_exit.assert_called_once()
    mock_other_state.on_entry.assert_called_once()
    mock_transition.get_actions.assert_called_once()
    assert valid_sm.get_current_state_id() == "other_state"


def test_no_valid_transition(valid_sm, mock_event, mock_transition, mock_initial_state):
    # Configure initial state correctly first
    mock_initial_state.get_id.return_value = "initial_state"
    # Then configure transition's source state
    source_state = Mock(spec=AbstractState)
    source_state.get_id.return_value = "non_matching_state"
    mock_transition.get_source_state.return_value = source_state

    valid_sm.start()
    current = valid_sm.get_current_state_id()
    valid_sm.process_event(mock_event)
    assert valid_sm.get_current_state_id() == current


def test_guard_raises_exception(valid_sm, mock_event, mock_transition, caplog):
    def raise_exception(*args, **kwargs):
        raise RuntimeError("Guard error")

    mock_guard = Mock()
    mock_guard.check.side_effect = raise_exception
    mock_transition.get_guard.return_value = mock_guard

    valid_sm.start()
    valid_sm.process_event(mock_event)
    # Should skip transition and log error with the new format
    expected_msg = (
        "Guard check failed for transition from 'initial_state' to 'other_state' in state 'initial_state': Guard error"
    )
    assert expected_msg in caplog.text


def test_multiple_valid_transitions(valid_sm, mock_event, mock_initial_state, mock_other_state):
    # Create second transition with higher priority
    t2 = Mock(spec=AbstractTransition)
    t2.get_source_state.return_value = mock_initial_state
    t2.get_target_state.return_value = mock_other_state
    t2.get_guard.return_value = None
    t2.get_actions.return_value = []
    t2.get_priority.return_value = 10

    valid_sm._transitions.append(t2)
    valid_sm.start()
    valid_sm.process_event(mock_event)
    # Should choose transition with priority=10
    t2.get_actions.assert_called()


def test_transition_action_exception(valid_sm, mock_event, mock_transition, caplog):
    def action_execute(*args, **kwargs):
        raise RuntimeError("Action failed")

    action = Mock()
    action.execute.side_effect = action_execute
    mock_transition.get_actions.return_value = [action]

    valid_sm.start()
    with pytest.raises(RuntimeError, match="Action failed"):
        valid_sm.process_event(mock_event)
    assert "Error during operation 'Transition'" in caplog.text


def test_hooks_on_enter_exit(valid_sm, mock_initial_state, mock_event):
    # Hook calls are indirectly tested but let's confirm they are called
    with (
        patch.object(valid_sm._hook_manager, "call_on_enter") as mock_call_on_enter,
        patch.object(valid_sm._hook_manager, "call_on_exit") as mock_call_on_exit,
    ):
        valid_sm.start()
        mock_call_on_enter.assert_called_with("initial_state")
        valid_sm.stop()
        # on_exit calls will be triggered on stop
        mock_call_on_exit.assert_called()


@patch("hsm.core.state_machine.Validator")
def test_composite_state_history_setting(mock_validator):
    """Test that history state is set when required"""
    validator_instance = mock_validator.return_value
    validator_instance.validate_structure.return_value = []
    validator_instance.validate_behavior.return_value = []

    # Setup states
    parent_state = Mock(spec=CompositeState)
    parent_state.get_id.return_value = "parent"
    parent_state.has_history.return_value = True
    parent_state._current_substate = Mock()
    parent_state._current_substate.get_id.return_value = "substate"
    parent_state._enter_substate = Mock()

    # Setup transition
    transition = Mock()
    transition.get_source_state.return_value = parent_state
    transition.get_target_state.return_value = parent_state
    transition.get_guard.return_value = None
    transition.get_actions.return_value = []

    # Include both parent_state and its substate in the list
    sm = StateMachine([parent_state, parent_state._current_substate], [transition], parent_state)

    sm.start()

    parent_state.set_history_state.assert_called_once_with(parent_state._current_substate)


@patch("hsm.core.state_machine.Validator")
def test_composite_state_no_history(mock_validator):
    """Test that history state is not set when not supported"""
    validator_instance = mock_validator.return_value
    validator_instance.validate_structure.return_value = []
    validator_instance.validate_behavior.return_value = []

    parent_state = Mock()
    parent_state.get_id.return_value = "parent"
    parent_state.has_history.return_value = False
    parent_state._current_substate = Mock()
    parent_state._current_substate.get_id.return_value = "substate"

    transition = Mock()
    transition.get_source_state.return_value = parent_state
    transition.get_target_state.return_value = parent_state
    transition.get_guard.return_value = None
    transition.get_actions.return_value = []

    sm = StateMachine([parent_state, parent_state._current_substate], [transition], parent_state)
    sm.start()

    parent_state.set_history_state.assert_not_called()


@patch("hsm.core.state_machine.Validator")
def test_transition_priority(mock_validator):
    """Test that highest priority transition is selected"""
    validator_instance = mock_validator.return_value
    validator_instance.validate_structure.return_value = []
    validator_instance.validate_behavior.return_value = []

    mock_state = Mock()
    mock_state.get_id.return_value = "mock_state_id"

    t1 = Mock()
    t1.get_priority.return_value = 1
    t1.get_source_state.return_value = mock_state
    t1.get_target_state.return_value = mock_state
    t1.get_guard.return_value = None
    t1.get_actions.return_value = []

    t2 = Mock()
    t2.get_priority.return_value = 2
    t2.get_source_state.return_value = mock_state
    t2.get_target_state.return_value = mock_state
    t2.get_guard.return_value = None
    t2.get_actions.return_value = []

    sm = StateMachine([mock_state], [t1, t2], mock_state)
    sm.start()

    event = Mock()
    event.get_id.return_value = "test_event"

    transition = sm._find_valid_transition(event)
    assert transition == t2


def test_state_data_isolation(valid_sm: StateMachine) -> None:
    """Test that each state maintains isolated data"""
    valid_sm.start()

    # Add data to initial state
    with valid_sm._data_manager.access_data() as data:
        data[valid_sm._initial_state.get_id()]["key1"] = "value1"
        data[valid_sm._transitions[0].get_target_state().get_id()]["key2"] = "value2"

    # Verify data isolation
    with valid_sm._data_manager.access_data() as data:
        assert data[valid_sm._initial_state.get_id()]["key1"] == "value1"
        assert "key2" not in data[valid_sm._initial_state.get_id()]
        target_id = valid_sm._transitions[0].get_target_state().get_id()
        assert data[target_id]["key2"] == "value2"
        assert "key1" not in data[target_id]


def test_data_access_during_transition(valid_sm: StateMachine, mock_event: Any, mock_transition: Any) -> None:
    """Test data access during transition execution"""
    action = Mock()
    mock_transition.get_actions.return_value = [action]

    valid_sm.start()

    with valid_sm._data_manager.access_data() as data:
        data[valid_sm._initial_state.get_id()]["test_key"] = "test_value"

    valid_sm.process_event(mock_event)

    action.execute.assert_called_once()
    kwargs = action.execute.call_args.kwargs
    assert "test_key" in kwargs["data"]
    assert kwargs["data"]["test_key"] == "test_value"


def test_transition_to_self(valid_sm, mock_initial_state):
    """Test state transitioning to itself"""
    self_transition = Mock()
    self_transition.get_source_state.return_value = mock_initial_state
    self_transition.get_target_state.return_value = mock_initial_state
    self_transition.get_guard.return_value = None
    self_transition.get_actions.return_value = []
    self_transition.get_priority.return_value = 1

    mock_initial_state.__eq__ = lambda self, other: other.get_id() == "initial_state"

    valid_sm._transitions = [self_transition]
    valid_sm.start()

    event = Mock()
    event.get_id.return_value = "self_transition_event"

    valid_sm.process_event(event)

    mock_initial_state


def test_hook_error_handling(valid_sm, mock_event, caplog):
    """Test that hook errors are caught and logged but don't stop execution"""

    def raise_error(*args):
        raise RuntimeError("Hook error")

    with patch.object(valid_sm._hook_manager, "call_on_enter", side_effect=raise_error):
        valid_sm.start()  # Should continue despite hook error
        assert "Hook error during start" in caplog.text
        assert valid_sm._running is True


def test_entry_action_error(valid_sm, mock_initial_state):
    """Test handling of errors during state entry"""
    mock_initial_state.on_entry.side_effect = RuntimeError("Entry error")

    with pytest.raises(InvalidStateError, match="Failed to enter initial state"):
        valid_sm.start()
    assert valid_sm._running is False


def test_exit_action_error(valid_sm, mock_event, mock_transition, caplog):
    """Test handling of errors during state exit"""
    mock_initial_state = valid_sm._initial_state
    with patch.object(mock_initial_state, "on_exit", side_effect=RuntimeError("Exit error")):
        valid_sm.start()
        try:
            valid_sm.process_event(mock_event)
        except RuntimeError:
            pass  # Expected error

        assert "Error during operation 'Transition'" in caplog.text
        assert "Exit error" in caplog.text


def test_nested_composite_states(mock_validator):
    """Test transitions between deeply nested composite states"""
    # Create a hierarchy: root -> mid -> leaf
    leaf_state = Mock(spec=AbstractState)
    leaf_state.get_id.return_value = "leaf"

    mid_state = Mock(spec=CompositeState)
    mid_state.get_id.return_value = "mid"
    mid_state._current_substate = leaf_state
    mid_state.get_substates.return_value = [leaf_state]
    mid_state.has_history.return_value = False
    mid_state._enter_substate = Mock()

    root_state = Mock(spec=CompositeState)
    root_state.get_id.return_value = "root"
    root_state._current_substate = mid_state
    root_state.get_substates.return_value = [mid_state]
    root_state.has_history.return_value = False
    root_state._enter_substate = Mock()

    # Set parent relationships
    leaf_state._parent_state = mid_state
    mid_state._parent_state = root_state

    transition = Mock(spec=AbstractTransition)
    transition.get_source_state.return_value = leaf_state
    transition.get_target_state.return_value = root_state
    transition.get_guard.return_value = None
    transition.get_actions.return_value = []

    validator_instance = mock_validator.return_value
    validator_instance.validate_structure.return_value = []
    validator_instance.validate_behavior.return_value = []

    sm = StateMachine([root_state, mid_state, leaf_state], [transition], root_state)
    sm.start()

    # Verify proper initialization of nested states
    assert sm.get_current_state_id() == "leaf"

    # Process event to transition up to root
    event = Mock(spec=AbstractEvent)
    sm.process_event(event)

    # Verify proper exit sequence
    assert leaf_state.on_exit.called
    assert mid_state.on_exit.called
    assert root_state.on_entry.called


def test_find_common_ancestor_cycle_protection(test_sm):
    """Test that _find_common_ancestor handles cycles in state hierarchy"""
    state1 = Mock(spec=AbstractState)
    state1.get_id.return_value = "state1"
    state2 = Mock(spec=AbstractState)
    state2.get_id.return_value = "state2"
    state3 = Mock(spec=AbstractState)
    state3.get_id.return_value = "state3"

    # Create a cycle
    state1._parent_state = state2
    state2._parent_state = state3
    state3._parent_state = state1

    result = test_sm._find_common_ancestor(state1, state2)
    assert result in (state1, state2, state3)


def test_find_common_ancestor_different_branches(test_sm):
    """Test finding common ancestor in different hierarchy branches"""
    root = Mock(spec=CompositeState)
    root.get_id.return_value = "root"
    branch1_mid = Mock(spec=CompositeState)
    branch1_mid.get_id.return_value = "branch1_mid"
    branch1_leaf = Mock(spec=AbstractState)
    branch1_leaf.get_id.return_value = "branch1_leaf"
    branch2_mid = Mock(spec=CompositeState)
    branch2_mid.get_id.return_value = "branch2_mid"
    branch2_leaf = Mock(spec=AbstractState)
    branch2_leaf.get_id.return_value = "branch2_leaf"

    # Set up hierarchy
    branch1_leaf._parent_state = branch1_mid
    branch1_mid._parent_state = root
    branch2_leaf._parent_state = branch2_mid
    branch2_mid._parent_state = root

    result = test_sm._find_common_ancestor(branch1_leaf, branch2_leaf)
    assert result == root


def test_data_thread_safety(test_sm: StateMachine) -> None:
    """Test concurrent access to state data"""
    import threading
    import time

    test_sm.start()
    initial_state_id = test_sm._initial_state.get_id()

    # Initialize counter
    with test_sm._data_manager.access_data() as data:
        data[initial_state_id]["counter"] = 0

    def modify_data() -> None:
        with test_sm._data_manager.access_data() as data:
            current_value = data[initial_state_id].get("counter", 0)
            time.sleep(0.01)
            data[initial_state_id]["counter"] = current_value + 1

    threads = []
    for _ in range(5):
        t = threading.Thread(target=modify_data)
        t.daemon = True
        threads.append(t)

    for t in threads:
        t.start()
        time.sleep(0.02)

    for t in threads:
        t.join(timeout=1.0)

    with test_sm._data_manager.access_data() as data:
        assert data[initial_state_id]["counter"] == 5


def test_validation_behavior_errors(mock_state, mock_transition):
    """Test behavior validation failures"""
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []

        error_mock = Mock()
        error_mock.severity = ValidationSeverity.ERROR.name
        error_mock.message = "Invalid behavior"
        error_mock.context = {"detail": "Missing transition"}

        validator_instance.validate_behavior.return_value = [error_mock]

        with pytest.raises(ConfigurationError) as exc_info:
            StateMachine([mock_state], [mock_transition], mock_state)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        # Check that we got a behavior validation error
        assert "Invalid state machine behavior" in str(error)
        # The error message is just the basic message without context
        assert str(error) == "Invalid state machine behavior"


@pytest.fixture
def mock_validator():
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []
        yield MockValidator


def test_guard_evaluation_error():
    """Test handling of guard evaluation errors"""
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []

        mock_state = Mock(spec=AbstractState)
        mock_state.get_id.return_value = "state1"

        mock_guard = Mock()
        mock_guard.check.side_effect = Exception("Guard error")

        mock_transition = Mock(spec=AbstractTransition)
        mock_transition.get_source_state.return_value = mock_state
        mock_transition.get_guard.return_value = mock_guard

        sm = StateMachine([mock_state], [mock_transition], mock_state)
        sm.start()

        # Guard error should be logged but transition skipped
        event = Mock(spec=AbstractEvent)
        sm.process_event(event)
        assert sm.get_current_state_id() == "state1"


def test_action_execution_error():
    """Test handling of action execution errors"""
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []

        source_state = Mock(spec=AbstractState)
        source_state.get_id.return_value = "source"
        target_state = Mock(spec=AbstractState)
        target_state.get_id.return_value = "target"

        mock_action = Mock()
        mock_action.execute.side_effect = Exception("Action error")

        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = source_state
        transition.get_target_state.return_value = target_state
        transition.get_guard.return_value = None
        transition.get_actions.return_value = [mock_action]

        sm = StateMachine([source_state, target_state], [transition], source_state)
        sm.start()

        event = Mock(spec=AbstractEvent)
        with pytest.raises(Exception, match="Action error"):
            sm.process_event(event)


def test_composite_state_history():
    """Test composite state history functionality"""
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []

        substate = Mock(spec=AbstractState)
        substate.get_id.return_value = "substate"

        composite = Mock(spec=CompositeState)
        composite.get_id.return_value = "composite"
        composite.has_history.return_value = True
        composite.get_substates.return_value = [substate]
        composite._current_substate = substate

        substate._parent_state = composite

        # Add a self-transition to satisfy the non-empty transitions requirement
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = composite
        transition.get_target_state.return_value = composite
        transition.get_guard.return_value = None
        transition.get_actions.return_value = []

        sm = StateMachine([composite, substate], [transition], composite)
        sm.start()

        # History should be set during initialization
        composite.set_history_state.assert_called_once_with(substate)


def test_invalid_transition_error():
    """Test handling of invalid transition attempts"""
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []

        state1 = Mock(spec=AbstractState)
        state1.get_id.return_value = "state1"
        state2 = Mock(spec=AbstractState)
        state2.get_id.return_value = "state2"

        # Create transition from state2->state1 when we're in state1
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = state2
        transition.get_target_state.return_value = state1

        sm = StateMachine([state1, state2], [transition], state1)
        sm.start()

        # No valid transition should be found
        event = Mock(spec=AbstractEvent)
        sm.process_event(event)
        assert sm.get_current_state_id() == "state1"


def test_duplicate_state_ids():
    """Test that duplicate state IDs are detected and raise an error"""
    state1 = Mock(spec=AbstractState)
    state1.get_id.return_value = "duplicate_id"
    state2 = Mock(spec=AbstractState)
    state2.get_id.return_value = "duplicate_id"

    transition = Mock(spec=AbstractTransition)
    transition.get_source_state.return_value = state1
    transition.get_target_state.return_value = state2

    with pytest.raises(ValueError, match="Duplicate state IDs found: {'duplicate_id'}"):
        StateMachine([state1, state2], [transition], state1)


def test_guard_check_failure_logging(caplog: LogCaptureFixture) -> None:
    """Test that guard check failures are logged with full context"""
    source_state = Mock(spec=AbstractState)
    source_state.get_id.return_value = "source"
    target_state = Mock(spec=AbstractState)
    target_state.get_id.return_value = "target"

    guard = Mock()
    guard.check.side_effect = RuntimeError("Guard check failed")

    transition = Mock(spec=AbstractTransition)
    transition.get_source_state.return_value = source_state
    transition.get_target_state.return_value = target_state
    transition.get_guard.return_value = guard
    transition.get_actions.return_value = []

    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []

        sm = StateMachine([source_state, target_state], [transition], source_state)
        sm.start()

        event = Mock(spec=AbstractEvent)
        sm.process_event(event)

        assert "Guard check failed for transition from 'source' to 'target' in state 'source'" in caplog.text


def test_initialize_state_data(valid_sm: StateMachine) -> None:
    """Test that state data is properly initialized for all states"""
    with valid_sm._data_manager.access_data() as data:
        # Should have empty dict for each state
        assert data[valid_sm._initial_state.get_id()] == {}
        target_id = valid_sm._transitions[0].get_target_state().get_id()
        assert data[target_id] == {}
        assert len(data) == 2  # Should only have data for our two states


def test_drill_down_no_substates(test_sm: StateMachine) -> None:
    """Test _drill_down with a non-composite state"""
    state = Mock(spec=AbstractState)
    state.get_id.return_value = "simple_state"
    
    with test_sm._data_manager.access_data() as data:
        data[state.get_id()] = {}
        result = test_sm._drill_down(state, data, None)
        assert result == state


def test_handle_operation_error_with_details(test_sm: StateMachine) -> None:
    """Test error handling with additional details"""
    error = RuntimeError("Test error")
    details = {"key": "value"}
    
    with (
        pytest.raises(RuntimeError, match="Test error"),
        patch.object(test_sm._logger, "error") as mock_error
    ):
        test_sm._handle_operation_error("TestOp", error, details)
        
    # Verify both error messages were logged with correct format
    mock_error.assert_has_calls([
        call("Error during operation '%s': %s", 'TestOp', error, exc_info=True),
        call("Error details: %s", details)
    ])


def test_exit_up_with_hook_error(test_sm: StateMachine) -> None:
    """Test _exit_up handles hook errors gracefully"""
    state = Mock(spec=AbstractState)
    state.get_id.return_value = "test_state"
    
    def raise_error(*args):
        raise RuntimeError("Hook error")
    
    with (
        patch.object(test_sm._hook_manager, "call_on_exit", side_effect=raise_error),
        patch.object(test_sm._logger, "warning") as mock_warning
    ):
        with test_sm._data_manager.access_data() as data:
            data[state.get_id()] = {}
            test_sm._exit_up(state, data)
            
        mock_warning.assert_called_with("Hook error during exit (continuing): Hook error")


def test_find_common_ancestor_same_state(test_sm: StateMachine) -> None:
    """Test _find_common_ancestor when both states are the same"""
    state = Mock(spec=AbstractState)
    state.get_id.return_value = "state"
    result = test_sm._find_common_ancestor(state, state)
    assert result == state


def test_find_common_ancestor_no_common(test_sm: StateMachine) -> None:
    """Test _find_common_ancestor when there is no common ancestor"""
    state1 = Mock(spec=AbstractState)
    state1.get_id.return_value = "state1"
    state1._parent_state = None
    
    state2 = Mock(spec=AbstractState)
    state2.get_id.return_value = "state2"
    state2._parent_state = None
    
    result = test_sm._find_common_ancestor(state1, state2)
    assert result is None


def test_drill_down_composite_no_substate(test_sm: StateMachine) -> None:
    """Test _drill_down with composite state that has no current substate"""
    composite = Mock(spec=CompositeState)
    composite.get_id.return_value = "composite"
    composite._current_substate = None
    
    with test_sm._data_manager.access_data() as data:
        data[composite.get_id()] = {}
        result = test_sm._drill_down(composite, data, None)
        assert result == composite


def test_enter_states_with_history(test_sm: StateMachine) -> None:
    """Test _enter_states with history-enabled composite state"""
    source = Mock(spec=AbstractState)
    source.get_id.return_value = "source"
    
    substate = Mock(spec=AbstractState)
    substate.get_id.return_value = "substate"
    
    target = Mock(spec=CompositeState)
    target.get_id.return_value = "target"
    target.has_history.return_value = True
    target._current_substate = substate
    target._enter_substate = Mock()
    
    event = Mock(spec=AbstractEvent)
    
    with test_sm._data_manager.access_data() as data:
        data[target.get_id()] = {}
        data[substate.get_id()] = {}
        test_sm._enter_states(source, target, event, data)
        
        target.set_history_state.assert_called_once_with(substate)


def test_exit_states_composite_target_substate(test_sm: StateMachine) -> None:
    """Test _exit_states when target is a substate of source"""
    # First create target so we can reference it
    target = Mock(spec=AbstractState)
    target.get_id.return_value = "target"
    target.on_exit = Mock()

    # Then create source with target as substate
    source = Mock(spec=CompositeState)
    source.get_id.return_value = "source"
    source._current_substate = target  # Set current substate
    source.get_substates.return_value = [target]
    source.on_exit = Mock()

    # Set up parent-child relationship
    target._parent_state = source

    # Set up state machine state
    test_sm._current_state = target
    test_sm._states = {
        source.get_id(): source,
        target.get_id(): target
    }

    event = Mock(spec=AbstractEvent)
    
    with test_sm._data_manager.access_data() as data:
        data[source.get_id()] = {}
        data[target.get_id()] = {}
        test_sm._exit_states(source, target, event, data)
        
        # Should not exit source state since target is its substate
        source.on_exit.assert_not_called()


def test_process_event_composite_state_substate_transition(test_sm: StateMachine) -> None:
    """Test processing event when transition is from a substate"""
    # Create states with proper hierarchy
    parent = Mock(spec=CompositeState)
    parent.get_id.return_value = "parent"
    parent.on_exit = Mock()
    
    substate = Mock(spec=AbstractState)
    substate.get_id.return_value = "substate"
    substate.on_exit = Mock()
    substate._parent_state = parent
    
    target = Mock(spec=AbstractState)
    target.get_id.return_value = "target"
    target.on_entry = Mock()
    target.on_exit = Mock()
    
    # Set up composite state
    parent._current_substate = substate
    parent.get_substates.return_value = [substate]
    
    # Set up transition
    transition = Mock(spec=AbstractTransition)
    transition.get_source_state.return_value = substate
    transition.get_target_state.return_value = target
    transition.get_guard.return_value = None
    transition.get_actions.return_value = []
    transition.get_priority.return_value = 1
    
    # Set up state machine
    test_sm._states = {
        parent.get_id(): parent,
        substate.get_id(): substate,
        target.get_id(): target
    }
    test_sm._transitions = [transition]
    test_sm._current_state = substate  # Set current state to substate, not parent
    test_sm._running = True
    
    # Initialize state data
    with test_sm._data_manager.access_data() as data:
        data[parent.get_id()] = {}
        data[substate.get_id()] = {}
        data[target.get_id()] = {}
    
    event = Mock(spec=AbstractEvent)
    test_sm.process_event(event)
    
    # Verify transition execution
    substate.on_exit.assert_called_once()
    target.on_entry.assert_called_once()


def test_initialize_state_data_composite_hierarchy(test_sm: StateMachine) -> None:
    """Test state data initialization with composite state hierarchy"""
    parent = Mock(spec=CompositeState)
    parent.get_id.return_value = "parent"
    
    child1 = Mock(spec=AbstractState)
    child1.get_id.return_value = "child1"
    child1._parent_state = parent
    
    child2 = Mock(spec=AbstractState)
    child2.get_id.return_value = "child2"
    child2._parent_state = parent
    
    parent.get_substates.return_value = [child1, child2]
    
    with patch("hsm.core.state_machine.Validator") as MockValidator:
        validator_instance = MockValidator.return_value
        validator_instance.validate_structure.return_value = []
        validator_instance.validate_behavior.return_value = []
        
        sm = StateMachine([parent, child1, child2], [Mock(spec=AbstractTransition)], parent)
        
        with sm._data_manager.access_data() as data:
            # Should have data dict for all states including parent and children
            assert data[parent.get_id()] == {}
            assert data[child1.get_id()] == {}
            assert data[child2.get_id()] == {}
            assert len(data) == 3


def test_rapid_fire_transitions(valid_sm: StateMachine, mock_event: Any, mock_transition: Any) -> None:
    """Test processing multiple events in rapid succession."""
    # Configure transitions to allow multiple transitions
    target_states = []
    transitions = []
    current_source = valid_sm._initial_state
    
    # Create a chain of 5 states with transitions between them
    for i in range(5):
        target = Mock(spec=AbstractState)
        target.get_id.return_value = f"state_{i}"
        target_states.append(target)
        
        trans = Mock(spec=AbstractTransition)
        trans.get_source_state.return_value = current_source
        trans.get_target_state.return_value = target
        trans.get_guard.return_value = None
        trans.get_actions.return_value = []
        trans.get_priority.return_value = 1
        transitions.append(trans)
        
        current_source = target
    
    # Clear existing transitions and states before adding new ones
    valid_sm._transitions = []
    valid_sm._states = {valid_sm._initial_state.get_id(): valid_sm._initial_state}
    
    valid_sm._states.update({state.get_id(): state for state in target_states})
    valid_sm._transitions.extend(transitions)
    
    # Initialize state data for new states
    with valid_sm._data_manager.access_data() as data:
        for state in target_states:
            data[state.get_id()] = {}
    
    valid_sm.start()
    
    # Process events rapidly
    events = [Event("event_" + str(i)) for i in range(5)]
    for event in events:
        valid_sm.process_event(event)
    
    # Verify we ended up in the last state
    assert valid_sm.get_current_state_id() == "state_4"


def test_transition_no_guard_no_actions(valid_sm: StateMachine, mock_event: Any) -> None:
    """Test a transition with neither guard nor actions."""
    source = Mock(spec=AbstractState)
    source.get_id.return_value = "source"
    target = Mock(spec=AbstractState)
    target.get_id.return_value = "target"
    
    # Create minimal transition
    transition = Mock(spec=AbstractTransition)
    transition.get_source_state.return_value = source
    transition.get_target_state.return_value = target
    transition.get_guard.return_value = None
    transition.get_actions.return_value = None  # Explicitly None instead of empty list
    transition.get_priority.return_value = 1
    
    # Add states and transition to machine
    valid_sm._states.update({
        source.get_id(): source,
        target.get_id(): target
    })
    valid_sm._transitions = [transition]
    valid_sm._initial_state = source
    
    # Initialize state data
    with valid_sm._data_manager.access_data() as data:
        data[source.get_id()] = {}
        data[target.get_id()] = {}
    
    valid_sm.start()
    valid_sm.process_event(mock_event)
    
    # Verify transition occurred without errors
    assert valid_sm.get_current_state_id() == "target"


def test_multiple_sibling_substates_transitions(valid_sm: StateMachine) -> None:
    """Test transitions between multiple sibling substates within a composite state."""
    # Create sibling substates
    sub1 = Mock(spec=AbstractState)
    sub1.get_id.return_value = "sub1"
    sub2 = Mock(spec=AbstractState)
    sub2.get_id.return_value = "sub2"
    sub3 = Mock(spec=AbstractState)
    sub3.get_id.return_value = "sub3"
    
    # Create parent composite
    parent = Mock(spec=CompositeState)
    parent.get_id.return_value = "parent"
    parent.get_substates.return_value = [sub1, sub2, sub3]
    parent._current_substate = sub1
    parent.has_history.return_value = True
    parent._enter_substate = Mock()
    
    # Set parent relationships
    sub1._parent_state = parent
    sub2._parent_state = parent
    sub3._parent_state = parent
    
    # Create transitions between siblings
    t1 = Mock(spec=AbstractTransition)
    t1.get_source_state.return_value = sub1
    t1.get_target_state.return_value = sub2
    t1.get_guard.return_value = None
    t1.get_actions.return_value = []
    t1.get_priority.return_value = 1
    
    t2 = Mock(spec=AbstractTransition)
    t2.get_source_state.return_value = sub2
    t2.get_target_state.return_value = sub3
    t2.get_guard.return_value = None
    t2.get_actions.return_value = []
    t2.get_priority.return_value = 1
    
    # Clear existing state machine configuration
    valid_sm._transitions = [t1, t2]
    valid_sm._states = {
        parent.get_id(): parent,
        sub1.get_id(): sub1,
        sub2.get_id(): sub2,
        sub3.get_id(): sub3
    }
    valid_sm._initial_state = parent
    
    # Initialize state data
    with valid_sm._data_manager.access_data() as data:
        for state in [parent, sub1, sub2, sub3]:
            data[state.get_id()] = {}
    
    valid_sm.start()
    
    # Verify initial substate
    assert valid_sm.get_current_state_id() == "sub1"
    
    # Mock the parent's behavior for setting current substate
    def update_current_substate(new_state):
        parent._current_substate = new_state
    parent.set_history_state.side_effect = update_current_substate
    
    # Transition between siblings
    valid_sm.process_event(Event("next"))
    assert valid_sm.get_current_state_id() == "sub2"
    parent.set_history_state.assert_called_with(sub2)
    
    valid_sm.process_event(Event("next"))
    assert valid_sm.get_current_state_id() == "sub3"
    parent.set_history_state.assert_called_with(sub3)


def test_sibling_transition_with_common_ancestor(valid_sm: StateMachine) -> None:
    """Test transitions between sibling states with proper ancestor handling."""
    # Create sibling substates
    sub1 = Mock(spec=AbstractState)
    sub1.get_id.return_value = "sub1"
    sub2 = Mock(spec=AbstractState)
    sub2.get_id.return_value = "sub2"
    
    # Create parent composite
    parent = Mock(spec=CompositeState)
    parent.get_id.return_value = "parent"
    parent.get_substates.return_value = [sub1, sub2]
    parent._current_substate = sub1
    parent.has_history.return_value = True
    parent._enter_substate = Mock()
    
    # Set parent relationships
    sub1._parent_state = parent
    sub2._parent_state = parent
    
    # Create transition between siblings
    transition = Mock(spec=AbstractTransition)
    transition.get_source_state.return_value = sub1
    transition.get_target_state.return_value = sub2
    transition.get_guard.return_value = None
    transition.get_actions.return_value = []
    transition.get_priority.return_value = 1
    
    # Configure state machine
    valid_sm._transitions = [transition]
    valid_sm._states = {
        parent.get_id(): parent,
        sub1.get_id(): sub1,
        sub2.get_id(): sub2
    }
    valid_sm._initial_state = parent
    
    # Initialize state data
    with valid_sm._data_manager.access_data() as data:
        for state in [parent, sub1, sub2]:
            data[state.get_id()] = {}
    
    valid_sm.start()
    assert valid_sm.get_current_state_id() == "sub1"
    
    # Reset the mock to clear the initial set_history_state call
    parent.set_history_state.reset_mock()
    
    # Execute transition
    valid_sm.process_event(Event("next"))
    
    # Verify proper transition execution
    assert valid_sm.get_current_state_id() == "sub2"
    sub1.on_exit.assert_called_once()
    sub2.on_entry.assert_called_once()
    parent.set_history_state.assert_called_once_with(sub2)


def test_composite_state_reentry(valid_sm: StateMachine) -> None:
    """Test re-entering a composite state preserves history."""
    # Create composite state with substates
    sub1 = Mock(spec=AbstractState)
    sub1.get_id.return_value = "sub1"
    sub2 = Mock(spec=AbstractState)
    sub2.get_id.return_value = "sub2"
    
    composite = Mock(spec=CompositeState)
    composite.get_id.return_value = "composite"
    composite.get_substates.return_value = [sub1, sub2]
    composite._current_substate = sub1
    composite.has_history.return_value = True
    
    # Create external state
    external = Mock(spec=AbstractState)
    external.get_id.return_value = "external"
    
    # Set relationships
    sub1._parent_state = composite
    sub2._parent_state = composite
    
    # Create transitions
    t1 = Mock(spec=AbstractTransition)
    t1.get_source_state.return_value = sub1
    t1.get_target_state.return_value = sub2
    t1.get_guard.return_value = None
    t1.get_actions.return_value = []
    t1.get_priority.return_value = 1
    
    t2 = Mock(spec=AbstractTransition)
    t2.get_source_state.return_value = sub2
    t2.get_target_state.return_value = external
    t2.get_guard.return_value = None
    t2.get_actions.return_value = []
    t2.get_priority.return_value = 1
    
    t3 = Mock(spec=AbstractTransition)
    t3.get_source_state.return_value = external
    t3.get_target_state.return_value = composite
    t3.get_guard.return_value = None
    t3.get_actions.return_value = []
    t3.get_priority.return_value = 1
    
    # Configure state machine
    valid_sm._transitions = [t1, t2, t3]
    valid_sm._states = {
        composite.get_id(): composite,
        sub1.get_id(): sub1,
        sub2.get_id(): sub2,
        external.get_id(): external
    }
    valid_sm._initial_state = composite
    
    # Initialize state data
    with valid_sm._data_manager.access_data() as data:
        for state in [composite, sub1, sub2, external]:
            data[state.get_id()] = {}
    
    valid_sm.start()
    assert valid_sm.get_current_state_id() == "sub1"
    
    # Transition to sub2
    valid_sm.process_event(Event("to_sub2"))
    assert valid_sm.get_current_state_id() == "sub2"
    
    # Exit to external state
    valid_sm.process_event(Event("to_external"))
    assert valid_sm.get_current_state_id() == "external"
    
    # Re-enter composite state
    valid_sm.process_event(Event("to_composite"))
    # Should return to sub2 due to history
    assert valid_sm.get_current_state_id() == "sub2"
