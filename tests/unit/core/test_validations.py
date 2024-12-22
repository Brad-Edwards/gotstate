# tests/unit/test_validations.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

from unittest.mock import MagicMock, Mock

import pytest

from hsm.core.errors import ValidationError
from hsm.core.events import Event
from hsm.core.states import CompositeState, State
from hsm.core.transitions import Transition
from hsm.runtime.graph import StateGraph


def test_validator_init():
    """Test validator initialization."""
    from hsm.core.validations import Validator

    validator = Validator()
    assert hasattr(validator, "_rules_engine")


def test_validate_event_empty_name():
    """Test validation fails for event with empty name."""
    from hsm.core.validations import Validator

    validator = Validator()
    event = Event("")

    with pytest.raises(ValidationError, match="Event must have a name"):
        validator.validate_event(event)


def test_validate_transition_missing_states():
    """Test validation fails for transition with missing states."""
    from hsm.core.validations import Validator

    validator = Validator()
    transition = Transition(None, None)

    with pytest.raises(ValidationError, match="Transition must have a valid source and target state"):
        validator.validate_transition(transition)


def test_validate_transition_invalid_guard():
    """Test validation fails for transition with non-callable guard."""
    from hsm.core.validations import Validator

    validator = Validator()
    source = State("source")
    target = State("target")
    transition = Transition(source, target, guards=["not_callable"])

    with pytest.raises(ValidationError, match="Transition guards must be callable"):
        validator.validate_transition(transition)


def test_validate_transition_invalid_action():
    """Test validation fails for transition with non-callable action."""
    from hsm.core.validations import Validator

    validator = Validator()
    source = State("source")
    target = State("target")
    transition = Transition(source, target, actions=["not_callable"])

    with pytest.raises(ValidationError, match="Transition actions must be callable"):
        validator.validate_transition(transition)


def test_validate_machine_no_initial_state():
    """Test validation fails for machine without initial state."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import Validator

    machine = MagicMock(spec=StateMachine)
    machine._mock_return_value = None
    machine._graph = MagicMock()
    machine._graph.get_initial_state.return_value = None
    machine.get_transitions.return_value = []
    machine.get_states.return_value = set()

    validator = Validator()
    with pytest.raises(ValidationError, match="StateMachine must have an initial state"):
        validator.validate_state_machine(machine)


def test_validate_machine_unreachable_states():
    """Test validation fails for machine with unreachable states."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import Validator

    machine = MagicMock(spec=StateMachine)
    machine._mock_return_value = None
    machine._graph = MagicMock()
    initial_state = State("initial")
    unreachable_state = State("unreachable")

    machine._graph.get_initial_state.return_value = initial_state
    machine._graph.get_children.return_value = []
    machine.get_transitions.return_value = []
    machine.get_states.return_value = {initial_state, unreachable_state}

    validator = Validator()
    with pytest.raises(ValidationError, match="States.*are not reachable"):
        validator.validate_state_machine(machine)


def test_validate_composite_state_no_initial():
    """Test validation fails for composite state without initial state."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import Validator

    machine = MagicMock(spec=StateMachine)
    machine._mock_return_value = None
    machine._graph = MagicMock()
    composite = CompositeState("composite")

    machine._graph.get_initial_state.side_effect = [composite, None, None]
    machine._graph.get_children.return_value = []
    machine.get_transitions.return_value = []
    machine.get_states.return_value = {composite}

    validator = Validator()
    with pytest.raises(ValidationError, match="Composite state 'composite' has no initial state"):
        validator.validate_state_machine(machine)


def test_async_validator():
    """Test async validator basic functionality."""
    from hsm.core.validations import AsyncValidator

    validator = AsyncValidator()
    assert isinstance(validator, AsyncValidator)


@pytest.mark.asyncio
async def test_async_validator_validation():
    """Test async validator validation."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import AsyncValidator

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()
    machine._graph.get_initial_state.return_value = None
    machine._started = True
    machine._graph.validate.return_value = []

    validator = AsyncValidator()
    with pytest.raises(ValidationError):
        await validator.validate_state_machine(machine)


def test_validate_machine_with_mock():
    """Test validation skips for mocks with _mock_return_value."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import Validator

    machine = MagicMock(spec=StateMachine)
    machine._mock_return_value = "something"  # This should trigger skip

    validator = Validator()
    validator.validate_state_machine(machine)  # Should not raise


def test_validate_machine_non_validation_error():
    """Test that non-ValidationErrors are wrapped."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import Validator

    machine = MagicMock(spec=StateMachine)
    machine._mock_return_value = None
    machine._graph = MagicMock()
    machine._graph.get_initial_state.side_effect = RuntimeError("Unexpected error")

    validator = Validator()
    with pytest.raises(ValidationError, match="Validation failed: Unexpected error"):
        validator.validate_state_machine(machine)


def test_validate_transition_no_guards_or_actions():
    """Test transition validation with no guards or actions."""
    from hsm.core.validations import Validator

    source = State("source")
    target = State("target")
    transition = Transition(source, target, guards=None, actions=None)

    validator = Validator()
    validator.validate_transition(transition)  # Should not raise


@pytest.mark.asyncio
async def test_async_validator_with_graph_errors():
    """Test async validator when graph validation returns errors."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import AsyncValidator

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()
    machine._graph.get_initial_state.return_value = State("test")
    machine._started = True
    machine._graph.validate.return_value = ["Error 1", "Error 2"]

    validator = AsyncValidator()
    with pytest.raises(ValidationError) as exc:
        await validator.validate_state_machine(machine)
    assert str(exc.value) == "Error 1\nError 2"


def test_build_initial_reachable_set():
    """Test building the initial reachable set with composite states."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import _DefaultValidationRules

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()

    # Create state hierarchy
    root = CompositeState("root")
    child1 = CompositeState("child1")
    child2 = State("child2")

    # Mock get_children to return hierarchy
    machine._graph.get_children.side_effect = lambda state: {root: [child1], child1: [child2], child2: []}.get(
        state, []
    )

    # Mock get_initial_state for composite states
    machine._graph.get_initial_state.side_effect = lambda state: {root: child1, child1: child2, None: root}.get(state)

    reachable = _DefaultValidationRules._build_initial_reachable_set(root, machine)
    assert reachable == {root, child1, child2}


@pytest.mark.asyncio
async def test_async_validator_no_errors():
    """Test async validator with no validation errors."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import AsyncValidator

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()
    machine._graph.get_initial_state.return_value = State("test")
    machine._started = True
    machine._graph.validate.return_value = []  # No errors
    machine._graph.get_current_state.return_value = State("current")

    validator = AsyncValidator()
    await validator.validate_state_machine(machine)  # Should not raise


@pytest.mark.asyncio
async def test_async_validator_not_started():
    """Test async validator with machine not started."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import AsyncValidator

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()
    machine._graph.get_initial_state.return_value = State("test")
    machine._started = False  # Machine not started
    machine._graph.validate.return_value = []

    validator = AsyncValidator()
    await validator.validate_state_machine(machine)  # Should not raise


def test_validate_transition_states():
    """Test validation of transition states existence."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import _DefaultValidationRules

    machine = MagicMock(spec=StateMachine)
    source = State("source")
    target = State("target")
    transition = Transition(source, target)

    # Test with source state missing
    with pytest.raises(ValidationError, match="State source is referenced in transition but not in state machine"):
        _DefaultValidationRules._validate_transition_states([transition], {target})

    # Test with target state missing
    with pytest.raises(ValidationError, match="State target is referenced in transition but not in state machine"):
        _DefaultValidationRules._validate_transition_states([transition], {source})


def test_expand_reachable_states():
    """Test expanding reachable states through transitions."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import _DefaultValidationRules

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()

    s1 = State("s1")
    s2 = State("s2")
    s3 = State("s3")

    t1 = Transition(s1, s2)
    t2 = Transition(s2, s3)

    reachable = {s1}
    transitions = [t1, t2]

    machine._graph.get_children.return_value = []
    _DefaultValidationRules._expand_reachable_states(reachable, transitions, machine)

    assert reachable == {s1, s2, s3}


def test_validate_state_reachability_no_unreachable():
    """Test validation passes when all states are reachable."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import _DefaultValidationRules

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()

    states = {State("s1"), State("s2")}
    _DefaultValidationRules._validate_state_reachability(states, states, machine)  # Should not raise


def test_validate_composite_states_all_valid():
    """Test validation of composite states with valid initial states."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import _DefaultValidationRules

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()

    composite = CompositeState("composite")
    machine._graph.get_initial_state.return_value = State("initial")

    _DefaultValidationRules._validate_composite_states({composite}, machine)  # Should not raise


@pytest.mark.asyncio
async def test_async_validator_with_current_state():
    """Test async validator with current state check."""
    from hsm.core.state_machine import StateMachine
    from hsm.core.validations import AsyncValidator

    machine = MagicMock(spec=StateMachine)
    machine._graph = MagicMock()
    machine._graph.get_initial_state.return_value = State("test")
    machine._started = True
    machine._graph.get_current_state.return_value = None
    machine._graph.validate.return_value = []

    validator = AsyncValidator()
    with pytest.raises(ValidationError, match="State machine must have a current state"):
        await validator.validate_state_machine(machine)
