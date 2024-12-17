# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import Any, Dict, List, Optional

import pytest

from hsm.core.transitions import Transition
from hsm.interfaces.abc import AbstractAction, AbstractGuard
from hsm.interfaces.protocols import Event
from hsm.interfaces.types import StateID

# -----------------------------------------------------------------------------
# MOCK IMPLEMENTATIONS FOR PROTOCOL TESTING
# -----------------------------------------------------------------------------


class MockGuard(AbstractGuard):
    def __init__(self, return_value: bool = True):
        self.return_value = return_value
        self.check_called = False
        self.last_event = None
        self.last_state_data = None

    def check(self, event: Event, state_data: Any) -> bool:
        self.check_called = True
        self.last_event = event
        self.last_state_data = state_data
        return self.return_value


class MockAction(AbstractAction):
    def __init__(self, should_raise: bool = False):
        self.execute_called = False
        self.last_event = None
        self.last_state_data = None
        self.should_raise = should_raise

    def execute(self, event: Event, state_data: Any) -> None:
        self.execute_called = True
        self.last_event = event
        self.last_state_data = state_data
        if self.should_raise:
            raise RuntimeError("Action execution failed")


class MockEvent(Event):
    def __init__(self, event_id: str, payload: Any = None, priority: int = 0):
        self._id = event_id
        self._payload = payload
        self._priority = priority

    def get_id(self) -> str:
        return self._id

    def get_payload(self) -> Any:
        return self._payload

    def get_priority(self) -> int:
        return self._priority


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def basic_transition() -> Transition:
    return Transition("source", "target")


@pytest.fixture
def guarded_transition() -> Transition:
    return Transition("source", "target", guard=MockGuard())


@pytest.fixture
def action_transition() -> Transition:
    return Transition("source", "target", actions=[MockAction()])


@pytest.fixture
def complex_transition() -> Transition:
    return Transition("source", "target", guard=MockGuard(), actions=[MockAction(), MockAction()], priority=10)


@pytest.fixture
def mock_event() -> MockEvent:
    return MockEvent("test_event", {"data": "value"}, 1)


# -----------------------------------------------------------------------------
# BASIC TRANSITION TESTS
# -----------------------------------------------------------------------------


def test_transition_initialization(basic_transition: Transition) -> None:
    assert basic_transition.get_source_state_id() == "source"
    assert basic_transition.get_target_state_id() == "target"
    assert basic_transition.get_guard() is None
    assert basic_transition.get_actions() == []
    assert basic_transition.get_priority() == 0


def test_transition_with_guard(guarded_transition: Transition) -> None:
    guard = guarded_transition.get_guard()
    assert isinstance(guard, MockGuard)
    assert not guard.check_called


def test_transition_with_actions(action_transition: Transition) -> None:
    actions = action_transition.get_actions()
    assert len(actions) == 1
    assert isinstance(actions[0], MockAction)
    assert not actions[0].execute_called


def test_complex_transition_attributes(complex_transition: Transition) -> None:
    assert complex_transition.get_source_state_id() == "source"
    assert complex_transition.get_target_state_id() == "target"
    assert isinstance(complex_transition.get_guard(), MockGuard)
    assert len(complex_transition.get_actions()) == 2
    assert complex_transition.get_priority() == 10


# -----------------------------------------------------------------------------
# GUARD BEHAVIOR TESTS
# -----------------------------------------------------------------------------


def test_guard_evaluation(guarded_transition: Transition, mock_event: MockEvent) -> None:
    guard = guarded_transition.get_guard()
    assert isinstance(guard, MockGuard)

    state_data = {"test": "data"}
    result = guard.check(mock_event, state_data)

    assert result is True
    assert guard.check_called
    assert guard.last_event == mock_event
    assert guard.last_state_data == state_data


def test_failing_guard(mock_event: MockEvent) -> None:
    failing_guard = MockGuard(return_value=False)
    Transition("source", "target", guard=failing_guard)

    state_data = {"test": "data"}
    result = failing_guard.check(mock_event, state_data)

    assert result is False
    assert failing_guard.check_called
    assert failing_guard.last_event == mock_event
    assert failing_guard.last_state_data == state_data


# -----------------------------------------------------------------------------
# ACTION EXECUTION TESTS
# -----------------------------------------------------------------------------


def test_action_execution(action_transition: Transition, mock_event: MockEvent) -> None:
    actions = action_transition.get_actions()
    action = actions[0]

    state_data = {"test": "data"}
    action.execute(mock_event, state_data)

    assert action.execute_called
    assert action.last_event == mock_event
    assert action.last_state_data == state_data


def test_multiple_actions_execution(complex_transition: Transition, mock_event: MockEvent) -> None:
    actions = complex_transition.get_actions()
    state_data = {"test": "data"}

    for action in actions:
        action.execute(mock_event, state_data)
        assert action.execute_called
        assert action.last_event == mock_event
        assert action.last_state_data == state_data


def test_failing_action(mock_event: MockEvent) -> None:
    failing_action = MockAction(should_raise=True)
    Transition("source", "target", actions=[failing_action])

    with pytest.raises(RuntimeError) as exc_info:
        failing_action.execute(mock_event, {"test": "data"})

    assert str(exc_info.value) == "Action execution failed"
    assert failing_action.execute_called


# -----------------------------------------------------------------------------
# EDGE CASES AND VALIDATION
# -----------------------------------------------------------------------------


def test_empty_state_ids() -> None:
    with pytest.raises(ValueError):
        Transition("", "target")

    with pytest.raises(ValueError):
        Transition("source", "")


def test_none_state_ids() -> None:
    with pytest.raises(TypeError):
        Transition(None, "target")  # type: ignore

    with pytest.raises(TypeError):
        Transition("source", None)  # type: ignore


def test_whitespace_state_ids() -> None:
    with pytest.raises(ValueError):
        Transition("  ", "target")

    with pytest.raises(ValueError):
        Transition("source", "  ")


def test_priority_bounds() -> None:
    transition = Transition("source", "target", priority=-1)
    assert transition.get_priority() == -1

    transition = Transition("source", "target", priority=1000000)
    assert transition.get_priority() == 1000000


# -----------------------------------------------------------------------------
# PROTOCOL COMPLIANCE TESTS
# -----------------------------------------------------------------------------


def test_transition_protocol_compliance(basic_transition: Transition) -> None:
    from hsm.interfaces.abc import AbstractTransition

    assert isinstance(basic_transition, AbstractTransition)


def test_guard_protocol_compliance() -> None:
    guard = MockGuard()
    from hsm.interfaces.abc import AbstractGuard

    assert isinstance(guard, AbstractGuard)


def test_action_protocol_compliance() -> None:
    action = MockAction()
    from hsm.interfaces.abc import AbstractAction

    assert isinstance(action, AbstractAction)


def test_transition_repr() -> None:
    transition = Transition("source", "target", guard=MockGuard(), actions=[MockAction(), MockAction()], priority=5)
    expected = "Transition(source='source', target='target', priority=5, guard=present, actions=2)"
    assert repr(transition) == expected
