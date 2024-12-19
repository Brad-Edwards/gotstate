import unittest
from unittest.mock import MagicMock

from hsm.core.states import CompositeState, State
from hsm.interfaces.abc import AbstractCompositeState, AbstractEvent, AbstractState
from hsm.interfaces.types import StateID


class MockEvent(AbstractEvent):
    """Mock implementation of AbstractEvent for testing purposes."""

    def __init__(self, name: str, data: dict = None):
        self._name = name
        self._data = data or {}

    def get_name(self) -> str:
        return self._name

    def get_data(self) -> dict:
        return self._data


class StateTests(unittest.TestCase):
    """Test cases for the State class."""

    def test_initialization_valid_id(self):
        """Test initialization with a valid state ID."""
        state = State("valid_id")
        self.assertEqual(state.get_id(), "valid_id")

    def test_initialization_none_id(self):
        """Test initialization with a None state ID."""
        with self.assertRaises(TypeError):
            State(None)

    def test_initialization_empty_id(self):
        """Test initialization with an empty state ID."""
        with self.assertRaises(ValueError):
            State("")

    def test_initialization_whitespace_id(self):
        """Test initialization with a whitespace-only state ID."""
        with self.assertRaises(ValueError):
            State("   ")

    def test_on_entry_not_implemented(self):
        """Test that on_entry raises NotImplementedError."""
        state = State("test_state")
        event = MockEvent("test_event")
        with self.assertRaises(NotImplementedError):
            state.on_entry(event, {})

    def test_on_exit_not_implemented(self):
        """Test that on_exit raises NotImplementedError."""
        state = State("test_state")
        event = MockEvent("test_event")
        with self.assertRaises(NotImplementedError):
            state.on_exit(event, {})

    def test_get_id_returns_correct_id(self):
        """Test that get_id returns the correct state ID."""
        state = State("test_state")
        self.assertEqual(state.get_id(), "test_state")

    def test_unicode_state_id(self):
        """Test that unicode characters in state_id are handled correctly."""
        state = State("état_🌟")
        self.assertEqual(state.get_id(), "état_🌟")

    def test_very_long_state_id(self):
        """Test that long state IDs are handled correctly."""
        long_id = "a" * 1000
        state = State(long_id)
        self.assertEqual(state.get_id(), long_id)

    def test_state_id_leading_trailing_spaces(self):
        """Test that leading/trailing spaces in state ID are preserved."""
        state = State(" test_state ")
        self.assertEqual(state.get_id(), " test_state ")


class CompositeStateTests(unittest.TestCase):
    """Test cases for the CompositeState class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.substate1 = MagicMock(spec=AbstractState)
        self.substate1.get_id.return_value = "substate1"
        self.substate2 = MagicMock(spec=AbstractState)
        self.substate2.get_id.return_value = "substate2"
        self.event = MockEvent("test_event")

    def test_initialization_valid(self):
        """Test valid initialization of a CompositeState."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], self.substate1)
        self.assertEqual(composite_state.get_id(), "composite")
        self.assertEqual(composite_state.get_substates(), [self.substate1, self.substate2])
        self.assertEqual(composite_state.get_initial_state(), self.substate1)
        self.assertFalse(composite_state.has_history())

    def test_initialization_empty_substates(self):
        """Test initialization with an empty substates list."""
        composite_state = CompositeState("composite", [])
        self.assertEqual(composite_state.get_substates(), [])
        with self.assertRaises(ValueError):
            composite_state.get_initial_state()

    def test_initialization_substates_not_list(self):
        """Test that initialization raises ValueError if substates is not a list."""
        with self.assertRaises(ValueError):
            CompositeState("composite", "not_a_list", self.substate1)  # type: ignore

    def test_initialization_initial_state_not_in_substates(self):
        """Test that initialization raises ValueError if initial_state is not in substates."""
        substate3 = MagicMock(spec=AbstractState)
        with self.assertRaises(ValueError):
            CompositeState("composite", [self.substate1, self.substate2], substate3)

    def test_initialization_initial_state_not_none_no_substates(self):
        """Test that initialization raises ValueError if initial_state is not None and no substates."""
        with self.assertRaises(ValueError):
            CompositeState("composite", [], self.substate1)

    def test_initialization_no_initial_state_with_substates(self):
        """Test that initial_state is set to the first substate when not provided with substates"""
        composite_state = CompositeState("composite", [self.substate1, self.substate2])
        self.assertEqual(composite_state.get_initial_state(), self.substate1)

    def test_on_entry_calls_initial_substate_on_entry(self):
        """Test that on_entry calls the initial substate's on_entry."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], self.substate1)
        composite_state.on_entry(self.event, {})
        self.substate1.on_entry.assert_called_once_with(self.event, {})

    def test_on_exit_calls_current_substate_on_exit(self):
        """Test that on_exit calls the current substate's on_exit."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], self.substate1)
        composite_state.on_entry(self.event, {})  # Enter initial substate
        composite_state.on_exit(self.event, {})
        self.substate1.on_exit.assert_called_once_with(self.event, {})

    def test_has_history_false_by_default(self):
        """Test that has_history is False by default."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2])
        self.assertFalse(composite_state.has_history())

    def test_has_history_true_when_set(self):
        """Test that has_history is True when set."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], has_history=True)
        self.assertTrue(composite_state.has_history())

    def test_set_history_state_valid(self):
        """Test setting the history state with a valid substate."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], has_history=True)
        composite_state.set_history_state(self.substate1)
        composite_state.on_entry(self.event, {})
        self.substate1.on_entry.assert_called_once_with(self.event, {})

    def test_set_history_state_invalid(self):
        """Test that set_history_state raises ValueError with an invalid substate."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], has_history=True)
        substate3 = MagicMock(spec=AbstractState)
        with self.assertRaises(ValueError):
            composite_state.set_history_state(substate3)

    def test_on_entry_with_history(self):
        """Test that on_entry enters the history state when has_history is True."""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], has_history=True)
        composite_state.on_entry(self.event, {})  # Enter initial substate (substate1)
        composite_state.on_exit(self.event, {})  # Exit substate1
        self.substate1.on_exit.assert_called_once_with(self.event, {})
        composite_state.on_entry(self.event, {})  # Re-enter, should go to history (substate1)
        self.substate1.on_entry.assert_called_with(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate1)

    def test_on_exit_sets_history_state(self):
        """Test on_exit sets the history state when has_history is True"""
        composite_state = CompositeState("composite", [self.substate1, self.substate2], has_history=True)
        composite_state.on_entry(self.event, {})
        composite_state.on_exit(self.event, {})
        self.assertEqual(composite_state._history_state, self.substate1)

    def test_parent_state_setter(self):
        """Test the parent_state setter property"""
        composite_state = CompositeState("composite", [self.substate1, self.substate2])
        mock_parent_state = MagicMock(spec=AbstractState)
        composite_state.parent_state = mock_parent_state
        self.assertEqual(composite_state.parent_state, mock_parent_state)

    def test_inheritance_order(self):
        """Test that CompositeState inherits correctly from both State and AbstractCompositeState."""
        composite_state = CompositeState("test", [])
        self.assertIsInstance(composite_state, State)
        self.assertIsInstance(composite_state, AbstractCompositeState)

    def test_state_methods_inherited(self):
        """Test that CompositeState inherits and can access State methods."""
        composite_state = CompositeState("test", [])
        self.assertEqual(composite_state.get_id(), "test")

    def test_composite_state_methods_inherited(self):
        """Test that CompositeState inherits and can access AbstractCompositeState methods."""
        substate = MagicMock(spec=AbstractState)
        composite_state = CompositeState("test", [substate])
        self.assertEqual(composite_state.get_substates(), [substate])

    def test_current_substate_after_entry(self):
        """Test that _current_substate is properly set after entry."""
        composite_state = CompositeState("test", [self.substate1, self.substate2])
        composite_state.on_entry(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate1)

    def test_current_substate_after_exit(self):
        """Test that _current_substate remains set after exit."""
        composite_state = CompositeState("test", [self.substate1, self.substate2])
        composite_state.on_entry(self.event, {})
        composite_state.on_exit(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate1)

    def test_history_state_cleared_on_entry_without_history(self):
        """Test that history state is not used when has_history is False."""
        composite_state = CompositeState("test", [self.substate1, self.substate2])
        composite_state._history_state = self.substate2  # manually set history
        composite_state.on_entry(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate1)  # should use initial state

    def test_parent_state_none_by_default(self):
        """Test that parent_state is None by default."""
        composite_state = CompositeState("test", [self.substate1])
        self.assertIsNone(composite_state.parent_state)

    def test_nested_composite_states(self):
        """Test that composite states can be nested."""
        inner_composite = CompositeState("inner", [self.substate1])
        outer_composite = CompositeState("outer", [inner_composite])
        self.assertIn(inner_composite, outer_composite.get_substates())
        self.assertEqual(inner_composite.parent_state, None)  # parent not set automatically

    def test_multiple_entry_exit_cycles(self):
        """Test multiple entry/exit cycles maintain correct state."""
        composite_state = CompositeState("test", [self.substate1, self.substate2], has_history=True)

        # First cycle
        composite_state.on_entry(self.event, {})
        composite_state.on_exit(self.event, {})

        # Second cycle
        composite_state.on_entry(self.event, {})
        composite_state.on_exit(self.event, {})

        self.substate1.on_entry.assert_called_with(self.event, {})
        self.assertEqual(self.substate1.on_entry.call_count, 2)
        self.assertEqual(self.substate1.on_exit.call_count, 2)

    def test_history_state_persists_after_multiple_transitions(self):
        """Test that history state persists correctly through multiple state transitions."""
        composite_state = CompositeState("test", [self.substate1, self.substate2], has_history=True)

        # Set history to substate2
        composite_state.set_history_state(self.substate2)

        # Multiple transitions
        composite_state.on_entry(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate2)
        composite_state.on_exit(self.event, {})
        composite_state.on_entry(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate2)

    def test_none_data_handling(self):
        """Test that None data is handled correctly in entry/exit."""
        composite_state = CompositeState("test", [self.substate1])
        composite_state.on_entry(self.event, None)
        self.substate1.on_entry.assert_called_with(self.event, None)

    def test_empty_event_handling(self):
        """Test handling of events with no data."""
        empty_event = MockEvent("empty")
        composite_state = CompositeState("test", [self.substate1])
        composite_state.on_entry(empty_event, {})
        self.substate1.on_entry.assert_called_with(empty_event, {})

    def test_substate_modification_after_initialization(self):
        """Test that substate list cannot be modified after initialization."""
        composite_state = CompositeState("test", [self.substate1])
        original_substates = composite_state.get_substates()

        # Attempt to modify the returned list
        substates = composite_state.get_substates()
        substates.append(self.substate2)

        # Verify the internal list wasn't modified
        self.assertEqual(composite_state.get_substates(), original_substates)

    def test_set_history_state_when_history_disabled(self):
        """Test that setting history state fails when history is disabled."""
        composite_state = CompositeState("test", [self.substate1], has_history=False)
        with self.assertRaises(ValueError):
            composite_state.set_history_state(self.substate1)

    def test_get_initial_state_after_modification(self):
        """Test that initial state remains consistent even if substate is modified."""
        composite_state = CompositeState("test", [self.substate1, self.substate2], self.substate1)
        initial_state = composite_state.get_initial_state()
        self.substate1.get_id.return_value = "modified"  # Modify the mock
        self.assertEqual(composite_state.get_initial_state(), initial_state)

    def test_deep_composite_state_hierarchy(self):
        """Test deep nesting of composite states."""
        inner_most = CompositeState("inner", [self.substate1])
        middle = CompositeState("middle", [inner_most])
        outer = CompositeState("outer", [middle])

        # Test parent relationships
        middle.parent_state = outer
        inner_most.parent_state = middle

        self.assertEqual(inner_most.parent_state, middle)
        self.assertEqual(middle.parent_state, outer)
        self.assertIsNone(outer.parent_state)

    def test_concurrent_history_states(self):
        """Test that multiple composite states maintain separate history."""
        state1 = CompositeState("state1", [self.substate1, self.substate2], has_history=True)
        state2 = CompositeState("state2", [self.substate1, self.substate2], has_history=True)

        state1.set_history_state(self.substate1)
        state2.set_history_state(self.substate2)

        state1.on_entry(self.event, {})
        state2.on_entry(self.event, {})

        self.assertEqual(state1._current_substate, self.substate1)
        self.assertEqual(state2._current_substate, self.substate2)

    def test_invalid_state_id_types(self):
        """Test initialization with invalid state ID types."""
        invalid_ids = [123, 3.14, [], {}, True]
        for invalid_id in invalid_ids:
            with self.assertRaises((TypeError, ValueError)):
                CompositeState(invalid_id, [self.substate1])  # type: ignore

    def test_set_history_state_multiple_times(self):
        """Test setting history state multiple times."""
        composite_state = CompositeState("test", [self.substate1, self.substate2], has_history=True)
        composite_state.set_history_state(self.substate1)
        composite_state.set_history_state(self.substate2)
        composite_state.on_entry(self.event, {})
        self.assertEqual(composite_state._current_substate, self.substate2)

    def test_clear_history_state(self):
        """Test that history state can be cleared by setting to None."""
        composite_state = CompositeState("test", [self.substate1, self.substate2], has_history=True)
        composite_state.set_history_state(self.substate2)
        with self.assertRaises(ValueError):
            composite_state.set_history_state(None)  # type: ignore

    def test_substate_uniqueness(self):
        """Test that the same state cannot be added multiple times."""
        with self.assertRaises(ValueError):
            CompositeState("test", [self.substate1, self.substate1])

    def test_empty_string_with_spaces_state_id(self):
        """Test that state ID with only spaces is rejected."""
        with self.assertRaises(ValueError):
            CompositeState("   ", [self.substate1])

    def test_different_states_same_id(self):
        """Test that different states with the same ID are considered duplicates."""
        duplicate_state = MagicMock(spec=AbstractState)
        duplicate_state.get_id.return_value = "substate1"
        with self.assertRaises(ValueError):
            CompositeState("test", [self.substate1, duplicate_state])

    def test_initial_state_none_with_history(self):
        """Test behavior when initial_state is None but has_history is True."""
        composite_state = CompositeState("test", [self.substate1], has_history=True)
        self.assertEqual(composite_state.get_initial_state(), self.substate1)
        self.assertTrue(composite_state.has_history())

    def test_nested_composite_state_history(self):
        """Test history behavior with nested composite states."""
        inner_composite = CompositeState("inner", [self.substate1], has_history=True)
        outer_composite = CompositeState("outer", [inner_composite], has_history=True)

        # Set history states
        inner_composite.set_history_state(self.substate1)
        outer_composite.set_history_state(inner_composite)

        # Verify both history states are maintained independently
        self.assertEqual(inner_composite._history_state, self.substate1)
        self.assertEqual(outer_composite._history_state, inner_composite)

    def test_get_substates_returns_copy(self):
        """Test that modifications to returned substates don't affect internal state."""
        composite_state = CompositeState("test", [self.substate1])
        substates = composite_state.get_substates()

        # Modify the returned list
        substates.append(self.substate2)

        # Verify internal state is unchanged
        self.assertEqual(len(composite_state.get_substates()), 1)
        self.assertEqual(composite_state.get_substates()[0], self.substate1)


if __name__ == "__main__":
    unittest.main()
