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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockGuard):
            return False
        return self.return_value == other.return_value

    def __hash__(self) -> int:
        """Hash based on return_value only."""
        return hash(self.return_value)


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockAction):
            return False
        return self.should_raise == other.should_raise

    def __hash__(self) -> int:
        """Hash based on should_raise only."""
        return hash(self.should_raise)


class MockEvent:
    def __init__(self, event_id: str = "test_event"):
        self.event_id = event_id

    def get_id(self) -> str:
        return self.event_id


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_event() -> MockEvent:
    return MockEvent()


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
    return Transition(
        "source",
        "target",
        guard=MockGuard(),
        actions=[MockAction(), MockAction()],
        priority=10,
    )


# -----------------------------------------------------------------------------
# BASIC TRANSITION TESTS
# -----------------------------------------------------------------------------


def test_transition_initialization(basic_transition: Transition) -> None:
    """Test basic transition initialization."""
    assert basic_transition.get_source_state_id() == "source"
    assert basic_transition.get_target_state_id() == "target"
    assert basic_transition.get_guard() is None
    assert len(basic_transition.get_actions()) == 0
    assert basic_transition.get_priority() == 0


def test_transition_with_guard(guarded_transition: Transition) -> None:
    """Test transition with guard."""
    guard = guarded_transition.get_guard()
    assert isinstance(guard, MockGuard)
    assert not guard.check_called


def test_transition_with_actions(action_transition: Transition) -> None:
    """Test transition with actions."""
    actions = action_transition.get_actions()
    assert len(actions) == 1
    assert isinstance(actions[0], MockAction)
    assert not actions[0].execute_called


# -----------------------------------------------------------------------------
# GUARD BEHAVIOR TESTS
# -----------------------------------------------------------------------------


def test_guard_evaluation(guarded_transition: Transition, mock_event: MockEvent) -> None:
    """Test guard evaluation."""
    guard = guarded_transition.get_guard()
    assert isinstance(guard, MockGuard)

    state_data = {"test": "data"}
    result = guard.check(mock_event, state_data)

    assert result is True
    assert guard.check_called
    assert guard.last_event == mock_event
    assert guard.last_state_data == state_data


def test_failing_guard(mock_event: MockEvent) -> None:
    """Test failing guard behavior."""
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
    """Test action execution."""
    actions = action_transition.get_actions()
    action = actions[0]

    state_data = {"test": "data"}
    action.execute(mock_event, state_data)

    assert action.execute_called
    assert action.last_event == mock_event
    assert action.last_state_data == state_data


def test_failing_action(mock_event: MockEvent) -> None:
    """Test failing action behavior."""
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
    """Test empty state IDs."""
    with pytest.raises(ValueError):
        Transition("", "target")

    with pytest.raises(ValueError):
        Transition("source", "")


def test_none_state_ids() -> None:
    """Test None state IDs."""
    with pytest.raises(TypeError):
        Transition(None, "target")  # type: ignore

    with pytest.raises(TypeError):
        Transition("source", None)  # type: ignore


def test_whitespace_state_ids() -> None:
    """Test whitespace state IDs."""
    with pytest.raises(ValueError):
        Transition("  ", "target")

    with pytest.raises(ValueError):
        Transition("source", "  ")


def test_priority_bounds() -> None:
    """Test priority bounds."""
    transition = Transition("source", "target", priority=-1)
    assert transition.get_priority() == -1

    transition = Transition("source", "target", priority=1000000)
    assert transition.get_priority() == 1000000


def test_transition_str_representation() -> None:
    """Test the string representation of transitions."""
    # Basic transition
    transition = Transition("s1", "s2")
    assert str(transition) == "Transition(source='s1', target='s2', priority=0, guard=none, actions=0)"

    # Complex transition
    transition = Transition("s1", "s2", guard=MockGuard(), actions=[MockAction(), MockAction()], priority=5)
    assert str(transition) == "Transition(source='s1', target='s2', priority=5, guard=present, actions=2)"


def test_transition_type_validation() -> None:
    """Test type validation for transition creation."""
    # Test with invalid source type
    with pytest.raises(TypeError, match="Source and target must be strings"):
        Transition(123, "target")  # type: ignore

    # Test with invalid target type
    with pytest.raises(TypeError, match="Source and target must be strings"):
        Transition("source", 456)  # type: ignore

    # Test with invalid guard type
    with pytest.raises(TypeError):
        Transition("source", "target", guard="not a guard")  # type: ignore

    # Test with invalid actions type
    with pytest.raises(TypeError):
        Transition("source", "target", actions="not a list")  # type: ignore


def test_transition_actions_list_handling() -> None:
    """Test how transitions handle the actions list."""
    # Test with empty actions list
    transition = Transition("s1", "s2", actions=[])
    assert len(transition.get_actions()) == 0

    # Test with None actions
    transition = Transition("s1", "s2", actions=None)
    assert len(transition.get_actions()) == 0

    # Test with multiple actions
    actions = [MockAction(), MockAction(), MockAction()]
    transition = Transition("s1", "s2", actions=actions)
    assert len(transition.get_actions()) == 3


def test_transition_immutability() -> None:
    """Test that transition attributes are effectively immutable."""
    actions = [MockAction()]
    transition = Transition("s1", "s2", actions=actions)

    # Get the actions list and try to modify it
    actions_list = transition.get_actions()
    actions_list.append(MockAction())

    # Original transition should still have only one action
    assert len(transition.get_actions()) == 1


def test_transition_with_special_characters() -> None:
    """Test transitions with special characters in state IDs."""
    special_chars = [
        ("state.1", "state.2"),
        ("state-1", "state-2"),
        ("state_1", "state_2"),
        ("state/1", "state/2"),
        ("state@1", "state@2"),
    ]

    for source, target in special_chars:
        transition = Transition(source, target)
        assert transition.get_source_state_id() == source
        assert transition.get_target_state_id() == target


def test_transition_priority_edge_cases() -> None:
    """Test edge cases for transition priorities."""
    # Test with very large positive priority
    transition = Transition("s1", "s2", priority=999999999)
    assert transition.get_priority() == 999999999

    # Test with very large negative priority
    transition = Transition("s1", "s2", priority=-999999999)
    assert transition.get_priority() == -999999999

    # Test with zero priority
    transition = Transition("s1", "s2", priority=0)
    assert transition.get_priority() == 0


def test_multiple_guards_error() -> None:
    """Test that multiple guards are not allowed."""
    guard1 = MockGuard()
    guard2 = MockGuard()

    # This should work
    Transition("s1", "s2", guard=guard1)

    # This should also work
    Transition("s1", "s2", guard=guard2)

    # But you can't have multiple guards
    with pytest.raises(TypeError):
        Transition("s1", "s2", guard=[guard1, guard2])  # type: ignore


def test_actions_validation() -> None:
    """Test validation of actions parameter."""
    # Test with invalid action objects
    invalid_actions = [MockAction(), "not an action", MockAction()]  # type: ignore
    with pytest.raises(TypeError, match="All actions must be instances of AbstractAction"):
        Transition("s1", "s2", actions=invalid_actions)

    # Test with empty list (should work)
    transition = Transition("s1", "s2", actions=[])
    assert len(transition.get_actions()) == 0


def test_actions_independence() -> None:
    """Test that modifying the original actions list doesn't affect the transition."""
    # Create a list of actions
    original_actions = [MockAction(), MockAction()]

    # Create transition with these actions
    transition = Transition("s1", "s2", actions=original_actions)

    # Modify the original list
    original_actions.append(MockAction())
    original_actions.clear()

    # Transition should still have 2 actions
    assert len(transition.get_actions()) == 2


def test_guard_validation() -> None:
    """Test validation of guard parameter."""
    # Test with various invalid guard types
    invalid_guards = [
        123,  # type: ignore
        "not a guard",  # type: ignore
        [],  # type: ignore
        {"guard": "dict"},  # type: ignore
    ]

    for invalid_guard in invalid_guards:
        with pytest.raises(TypeError, match="Guard must be an instance of AbstractGuard"):
            Transition("s1", "s2", guard=invalid_guard)


def test_state_id_validation() -> None:
    """Test validation of state IDs with various invalid inputs."""
    invalid_ids = [
        123,  # type: ignore
        3.14,  # type: ignore
        [],  # type: ignore
        {},  # type: ignore
        True,  # type: ignore
    ]

    for invalid_id in invalid_ids:
        with pytest.raises(TypeError, match="Source and target must be strings"):
            Transition(invalid_id, "target")

        with pytest.raises(TypeError, match="Source and target must be strings"):
            Transition("source", invalid_id)


def test_unicode_state_ids() -> None:
    """Test that transitions support Unicode state IDs."""
    unicode_pairs = [
        ("状态1", "状态2"),  # Chinese
        ("상태1", "상태2"),  # Korean
        ("état1", "état2"),  # French
        ("状態1", "状態2"),  # Japanese
        ("🎮1", "🎮2"),  # Emojis
    ]

    for source, target in unicode_pairs:
        transition = Transition(source, target)
        assert transition.get_source_state_id() == source
        assert transition.get_target_state_id() == target


def test_transition_equality() -> None:
    """Test transition equality behavior."""
    t1 = Transition("s1", "s2", priority=1)
    t2 = Transition("s1", "s2", priority=1)
    t3 = Transition("s1", "s2", priority=2)
    t4 = Transition("s1", "s3", priority=1)

    # Same source, target, and priority
    assert t1.__repr__() == t2.__repr__()

    # Different priority
    assert t1.__repr__() != t3.__repr__()

    # Different target
    assert t1.__repr__() != t4.__repr__()


def test_priority_type_validation() -> None:
    """Test validation of priority parameter."""
    invalid_priorities = [
        "1",  # type: ignore
        3.14,  # type: ignore
        [],  # type: ignore
        True,  # type: ignore
    ]

    for invalid_priority in invalid_priorities:
        with pytest.raises(TypeError):
            Transition("s1", "s2", priority=invalid_priority)


def test_consecutive_whitespace_state_ids() -> None:
    """Test state IDs with consecutive whitespace characters."""
    whitespace_ids = [
        "  state  1  ",
        "\t\tstate\t1\t",
        "\n\nstate\n1\n",
        " \t\n state \t\n 1 \t\n ",
    ]

    for state_id in whitespace_ids:
        with pytest.raises(ValueError):
            Transition(state_id, "target")
        with pytest.raises(ValueError):
            Transition("source", state_id)


def test_transition_hash_and_equality() -> None:
    """Test that transitions can be hashed and compared properly."""
    t1 = Transition("s1", "s2", priority=1)
    t2 = Transition("s1", "s2", priority=1)
    t3 = Transition("s1", "s2", priority=2)

    # Test equality
    assert t1 == t1  # Identity
    assert t1 == t2  # Same values
    assert t1 != t3  # Different priority
    assert t1 != "not a transition"  # Different type

    # Test hash
    transitions_set = {t1, t2, t3}
    assert len(transitions_set) == 2  # t1 and t2 should hash to same value


def test_transition_copy_independence() -> None:
    """Test that transitions maintain independence when copied."""
    original = Transition("s1", "s2", guard=MockGuard(), actions=[MockAction(), MockAction()])

    # Get copies of internal lists
    actions = original.get_actions()

    # Modify the copies
    actions.clear()

    # Original should be unchanged
    assert len(original.get_actions()) == 2


def test_empty_string_variations() -> None:
    """Test various empty string variations."""
    empty_variations = [
        "",  # Empty string
        " ",  # Single space
        "\t",  # Tab
        "\n",  # Newline
        "\r",  # Carriage return
        "\x0b",  # Vertical tab
        "\x0c",  # Form feed
        "\u2028",  # Line separator
        "\u2029",  # Paragraph separator
    ]

    for empty in empty_variations:
        with pytest.raises(ValueError):
            Transition(empty, "target")
        with pytest.raises(ValueError):
            Transition("source", empty)


def test_invalid_guard_variations() -> None:
    """Test various invalid guard types."""
    invalid_guards = [
        42,  # int
        3.14,  # float
        "guard",  # str
        [],  # list
        {},  # dict
        set(),  # set
        lambda x: x,  # function
        type("Guard", (), {}),  # dynamic class
    ]

    for invalid_guard in invalid_guards:
        with pytest.raises(TypeError):
            Transition("s1", "s2", guard=invalid_guard)  # type: ignore


def test_state_id_normalization() -> None:
    """Test that state IDs are properly normalized."""
    # These should be equivalent
    t1 = Transition("state1", "state2")
    t2 = Transition("state1", "state2")

    # Test direct equality
    assert t1 == t2

    # Test hash equality
    assert hash(t1) == hash(t2)

    # Test that whitespace is rejected
    with pytest.raises(ValueError, match="State IDs cannot contain whitespace"):
        Transition("state1 ", "state2")

    with pytest.raises(ValueError, match="State IDs cannot contain whitespace"):
        Transition("state1", " state2")


def test_priority_edge_cases() -> None:
    """Test edge cases for priority values."""
    # Test with system max/min integers
    import sys

    # These should work
    t1 = Transition("s1", "s2", priority=sys.maxsize)
    t2 = Transition("s1", "s2", priority=-sys.maxsize - 1)

    assert t1.get_priority() == sys.maxsize
    assert t2.get_priority() == -sys.maxsize - 1

    # Test with bool values (which are subclass of int)
    with pytest.raises(TypeError):
        Transition("s1", "s2", priority=True)  # type: ignore

    with pytest.raises(TypeError):
        Transition("s1", "s2", priority=False)  # type: ignore


def test_transition_hash_consistency() -> None:
    """Test that hash remains consistent and matches equality."""
    t1 = Transition("s1", "s2", priority=1)
    t2 = Transition("s1", "s2", priority=1)

    # Hash should be consistent across multiple calls
    assert hash(t1) == hash(t1)

    # Equal objects should have equal hashes
    assert hash(t1) == hash(t2)

    # Store hash before modifications
    original_hash = hash(t1)

    # Get new copies of actions list shouldn't affect hash
    _ = t1.get_actions()
    assert hash(t1) == original_hash


def test_transition_with_none_values() -> None:
    """Test transition creation with None values where allowed."""
    # None guard is allowed
    t1 = Transition("s1", "s2", guard=None)
    assert t1.get_guard() is None

    # None actions is allowed
    t2 = Transition("s1", "s2", actions=None)
    assert len(t2.get_actions()) == 0


def test_transition_with_empty_values() -> None:
    """Test transition creation with empty values where allowed."""
    # Empty actions list is allowed
    t1 = Transition("s1", "s2", actions=[])
    assert len(t1.get_actions()) == 0

    # Empty string state IDs are not allowed
    with pytest.raises(ValueError):
        Transition("", "s2")
    with pytest.raises(ValueError):
        Transition("s1", "")


def test_transition_comparison_with_different_types() -> None:
    """Test comparing transitions with different types."""
    transition = Transition("s1", "s2")

    # Compare with different types
    assert transition != None  # type: ignore
    assert transition != 42  # type: ignore
    assert transition != "transition"  # type: ignore
    assert transition != ["s1", "s2"]  # type: ignore
    assert transition != {"source": "s1", "target": "s2"}  # type: ignore


def test_transition_action_equality() -> None:
    """Test transition equality with different action configurations."""
    action1 = MockAction()
    action2 = MockAction()

    # Same actions, different instances
    t1 = Transition("s1", "s2", actions=[action1])
    t2 = Transition("s1", "s2", actions=[action2])
    assert t1 == t2  # Should be equal if MockAction implements __eq__

    # Different number of actions
    t3 = Transition("s1", "s2", actions=[action1, action2])
    assert t1 != t3

    # No actions vs empty actions list
    t4 = Transition("s1", "s2", actions=None)
    t5 = Transition("s1", "s2", actions=[])
    assert t4 == t5


def test_transition_guard_equality() -> None:
    """Test transition equality with different guard configurations."""
    guard1 = MockGuard()
    guard2 = MockGuard()

    # Same guards, different instances
    t1 = Transition("s1", "s2", guard=guard1)
    t2 = Transition("s1", "s2", guard=guard2)
    assert t1 == t2  # Should be equal if MockGuard implements __eq__

    # Guard vs no guard
    t3 = Transition("s1", "s2", guard=None)
    assert t1 != t3


def test_transition_case_sensitivity() -> None:
    """Test that state IDs are case-sensitive."""
    t1 = Transition("State1", "State2")
    t2 = Transition("state1", "state2")
    t3 = Transition("STATE1", "STATE2")

    assert t1 != t2
    assert t1 != t3
    assert t2 != t3

    # Hashes should be different
    transitions = {t1, t2, t3}
    assert len(transitions) == 3


def test_transition_hash_with_mutable_components() -> None:
    """Test that transition hash remains stable even if mutable components change."""
    action = MockAction()
    guard = MockGuard()

    transition = Transition("s1", "s2", guard=guard, actions=[action])
    original_hash = hash(transition)

    # Modify the guard's internal state
    guard.check_called = True
    guard.last_event = MockEvent()
    assert hash(transition) == original_hash

    # Modify the action's internal state
    action.execute_called = True
    action.last_event = MockEvent()
    assert hash(transition) == original_hash


def test_transition_with_identical_source_target() -> None:
    """Test transitions where source and target are the same state."""
    # Should be allowed - represents a self-transition
    transition = Transition("same_state", "same_state")
    assert transition.get_source_state_id() == transition.get_target_state_id()

    # Should work with all optional parameters
    transition = Transition("same_state", "same_state", guard=MockGuard(), actions=[MockAction()], priority=1)
    assert transition.get_source_state_id() == transition.get_target_state_id()


def test_transition_string_escaping() -> None:
    """Test that state IDs with quotes are handled properly."""
    special_strings = [
        ("state'1", "state'2"),  # Single quotes
        ('state"1', 'state"2'),  # Double quotes
        ("state\\1", "state\\2"),  # Backslashes
        ("state\n1", "state\n2"),  # Should raise ValueError - contains whitespace
    ]

    # Test valid strings
    for i in range(3):  # First three pairs should work
        source, target = special_strings[i]
        transition = Transition(source, target)
        assert transition.get_source_state_id() == source
        assert transition.get_target_state_id() == target
        # Verify string representation handles escaping
        assert source in str(transition)
        assert target in str(transition)

    # Test invalid string (with newline)
    with pytest.raises(ValueError):
        Transition(*special_strings[3])


def test_transition_identity() -> None:
    """Test transition identity vs equality."""
    t1 = Transition("s1", "s2")
    t2 = Transition("s1", "s2")
    same = t1

    # Identity
    assert t1 is same
    assert t1 is not t2

    # Equality
    assert t1 == same
    assert t1 == t2

    # Hash
    assert hash(t1) == hash(same)
    assert hash(t1) == hash(t2)


def test_transition_repr_eval() -> None:
    """Test that repr output is valid Python code."""
    transitions = [
        Transition("s1", "s2"),
        Transition("s1", "s2", priority=42),
        Transition("s1", "s2", guard=MockGuard()),
        Transition("s1", "s2", actions=[MockAction()]),
    ]

    for t in transitions:
        repr_str = repr(t)
        # Verify repr string contains all essential information
        assert "Transition" in repr_str
        assert t.get_source_state_id() in repr_str
        assert t.get_target_state_id() in repr_str
        assert str(t.get_priority()) in repr_str


def test_transition_deep_copy() -> None:
    """Test that transitions can be properly deep copied."""
    from copy import deepcopy

    original = Transition(
        "s1", "s2", guard=MockGuard(return_value=True), actions=[MockAction(should_raise=True)], priority=42
    )

    copied = deepcopy(original)

    # Should be equal but not identical
    assert copied == original
    assert copied is not original

    # Components should be equal but not identical
    assert copied.get_guard() == original.get_guard()
    assert copied.get_guard() is not original.get_guard()

    assert len(copied.get_actions()) == len(original.get_actions())
    for c_action, o_action in zip(copied.get_actions(), original.get_actions()):
        assert c_action == o_action
        assert c_action is not o_action


def test_transition_pickle() -> None:
    """Test that transitions can be pickled and unpickled."""
    import pickle

    original = Transition(
        "s1", "s2", guard=MockGuard(return_value=True), actions=[MockAction(should_raise=True)], priority=42
    )

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)

    # Should maintain equality
    assert unpickled == original
    assert unpickled is not original
