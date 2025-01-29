"""Unit tests for the State class and its subclasses.

Tests the state hierarchy management and behavior.
"""

import unittest
from unittest.mock import Mock, patch
from gotstate.core.state import (
    State,
    StateType,
    CompositeState,
    PseudoState,
    HistoryState,
    ConnectionPointState,
    ChoiceState,
    JunctionState
)


class TestState(unittest.TestCase):
    """Test cases for the base State class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_id = "test_state"
        self.state_type = StateType.SIMPLE
        self.parent = None
        self.data = {"key": "value"}
        
    def test_state_creation(self):
        """Test that a State can be created with valid parameters."""
        state = State(
            state_id=self.state_id,
            state_type=self.state_type,
            parent=self.parent,
            data=self.data
        )
        
        self.assertEqual(state.id, self.state_id)
        self.assertEqual(state.type, self.state_type)
        self.assertIsNone(state.parent)
        self.assertEqual(state.data, self.data)
        self.assertFalse(state.is_active)
        self.assertEqual(len(state.children), 0)
        
    def test_state_invalid_id(self):
        """Test that State creation fails with invalid ID."""
        with self.assertRaises(ValueError):
            State(
                state_id=None,
                state_type=self.state_type
            )
            
        with self.assertRaises(ValueError):
            State(
                state_id="",
                state_type=self.state_type
            )
            
    def test_state_invalid_type(self):
        """Test that State creation fails with invalid type."""
        with self.assertRaises(ValueError):
            State(
                state_id=self.state_id,
                state_type=None
            )
            
        with self.assertRaises(ValueError):
            State(
                state_id=self.state_id,
                state_type="invalid"
            )
            
    def test_state_data_isolation(self):
        """Test that state data is properly isolated."""
        original_data = {"key": "value"}
        state = State(
            state_id=self.state_id,
            state_type=self.state_type,
            data=original_data
        )
        
        # Verify that modifying the original data doesn't affect the state
        original_data["key"] = "modified"
        self.assertEqual(state.data["key"], "value")
        
        # Verify that modifying the returned data doesn't affect the state
        state_data = state.data
        state_data["key"] = "modified"
        self.assertEqual(state.data["key"], "value")
        
    def test_state_activation(self):
        """Test state activation and deactivation."""
        state = State(
            state_id=self.state_id,
            state_type=self.state_type
        )
        
        self.assertFalse(state.is_active)
        state.activate()
        self.assertTrue(state.is_active)
        state.deactivate()
        self.assertFalse(state.is_active)
        
    def test_state_parent_child_relationship(self):
        """Test parent-child relationship management."""
        parent = State(
            state_id="parent",
            state_type=StateType.COMPOSITE
        )
        
        child = State(
            state_id="child",
            state_type=StateType.SIMPLE,
            parent=parent
        )
        
        self.assertEqual(child.parent, parent)
        self.assertIn(child.id, parent.children)
        self.assertEqual(parent.children[child.id], child)
        
        # Test child removal
        child.remove_from_parent()
        self.assertIsNone(child.parent)
        self.assertNotIn(child.id, parent.children)
        
    def test_state_cyclic_parent_prevention(self):
        """Test prevention of cyclic parent-child relationships."""
        state1 = State(
            state_id="state1",
            state_type=StateType.COMPOSITE
        )
        
        state2 = State(
            state_id="state2",
            state_type=StateType.COMPOSITE,
            parent=state1
        )
        
        # Attempt to create a cycle
        with self.assertRaises(ValueError):
            state1.set_parent(state2)
            
    def test_state_entry_exit_actions(self):
        """Test state entry and exit action execution."""
        entry_action = Mock()
        exit_action = Mock()
        
        state = State(
            state_id=self.state_id,
            state_type=self.state_type,
            entry_action=entry_action,
            exit_action=exit_action
        )
        
        # Test entry action
        state.enter()
        entry_action.assert_called_once()
        self.assertTrue(state.is_active)
        
        # Test exit action
        state.exit()
        exit_action.assert_called_once()
        self.assertFalse(state.is_active)
        
    def test_state_do_activity(self):
        """Test state do-activity execution."""
        do_activity = Mock()
        
        state = State(
            state_id=self.state_id,
            state_type=self.state_type,
            do_activity=do_activity
        )
        
        # Test do-activity start/stop
        state.enter()
        do_activity.start.assert_called_once()
        
        state.exit()
        do_activity.stop.assert_called_once()


class TestCompositeState(unittest.TestCase):
    """Test cases for the CompositeState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_id = "test_composite"
        self.data = {"key": "value"}
        
    def test_composite_state_creation(self):
        """Test that a CompositeState can be created with valid parameters."""
        state = CompositeState(
            state_id=self.state_id,
            data=self.data
        )
        
        self.assertEqual(state.id, self.state_id)
        self.assertEqual(state.type, StateType.COMPOSITE)
        self.assertEqual(state.data, self.data)
        self.assertFalse(state.is_active)
        self.assertEqual(len(state.children), 0)
        self.assertEqual(len(state.regions), 0)
        
    def test_composite_state_child_management(self):
        """Test composite state child management."""
        parent = CompositeState(state_id="parent")
        child1 = State(state_id="child1", state_type=StateType.SIMPLE, parent=parent)
        child2 = State(state_id="child2", state_type=StateType.SIMPLE, parent=parent)
        
        self.assertEqual(len(parent.children), 2)
        self.assertIn(child1.id, parent.children)
        self.assertIn(child2.id, parent.children)
        
        # Test child removal
        child1.remove_from_parent()
        self.assertEqual(len(parent.children), 1)
        self.assertNotIn(child1.id, parent.children)
        self.assertIn(child2.id, parent.children)
        
    def test_composite_state_region_management(self):
        """Test composite state region management."""
        state = CompositeState(state_id=self.state_id)
        
        # Add regions
        region1 = state.add_region("region1")
        region2 = state.add_region("region2")
        
        self.assertEqual(len(state.regions), 2)
        self.assertIn("region1", state.regions)
        self.assertIn("region2", state.regions)
        
        # Test region removal
        state.remove_region("region1")
        self.assertEqual(len(state.regions), 1)
        self.assertNotIn("region1", state.regions)
        self.assertIn("region2", state.regions)
        
    def test_composite_state_activation(self):
        """Test composite state activation with regions."""
        parent = CompositeState(state_id="parent")
        region = parent.add_region("region1")
        child = State(state_id="child", state_type=StateType.SIMPLE, parent=parent)
        
        # Test activation propagation
        parent.activate()
        self.assertTrue(parent.is_active)
        self.assertTrue(region.is_active)
        
        # Test deactivation propagation
        parent.deactivate()
        self.assertFalse(parent.is_active)
        self.assertFalse(region.is_active)
        
    def test_composite_state_entry_exit(self):
        """Test composite state entry and exit with regions."""
        parent = CompositeState(state_id="parent")
        region = parent.add_region("region1")
        
        # Mock region enter/exit methods
        region.enter = Mock()
        region.exit = Mock()
        
        # Test entry propagation
        parent.enter()
        region.enter.assert_called_once()
        self.assertTrue(parent.is_active)
        
        # Test exit propagation
        parent.exit()
        region.exit.assert_called_once()
        self.assertFalse(parent.is_active)
        
    def test_composite_state_validation(self):
        """Test composite state validation rules."""
        state = CompositeState(state_id=self.state_id)
        
        # Test duplicate region names
        state.add_region("region1")
        with self.assertRaises(ValueError):
            state.add_region("region1")
            
        # Test invalid region names
        with self.assertRaises(ValueError):
            state.add_region("")
            
        with self.assertRaises(ValueError):
            state.add_region(None)


class TestPseudoState(unittest.TestCase):
    """Test cases for the PseudoState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_id = "test_pseudo"
        self.state_type = StateType.INITIAL
        self.parent = CompositeState(state_id="parent")
        
    def test_pseudo_state_creation(self):
        """Test that a PseudoState can be created with valid parameters."""
        state = PseudoState(
            state_id=self.state_id,
            state_type=self.state_type,
            parent=self.parent
        )
        
        self.assertEqual(state.id, self.state_id)
        self.assertEqual(state.type, self.state_type)
        self.assertEqual(state.parent, self.parent)
        self.assertFalse(state.is_active)
        self.assertEqual(len(state.children), 0)
        
    def test_pseudo_state_type_validation(self):
        """Test that PseudoState enforces valid state types."""
        # Valid pseudostate types
        valid_types = [
            StateType.INITIAL,
            StateType.CHOICE,
            StateType.JUNCTION,
            StateType.SHALLOW_HISTORY,
            StateType.DEEP_HISTORY,
            StateType.ENTRY_POINT,
            StateType.EXIT_POINT,
            StateType.TERMINATE
        ]
        
        for state_type in valid_types:
            state = PseudoState(
                state_id=self.state_id,
                state_type=state_type,
                parent=self.parent
            )
            self.assertEqual(state.type, state_type)
            
        # Invalid pseudostate types
        invalid_types = [
            StateType.SIMPLE,
            StateType.COMPOSITE,
            StateType.SUBMACHINE,
            StateType.FINAL
        ]
        
        for state_type in invalid_types:
            with self.assertRaises(ValueError):
                PseudoState(
                    state_id=self.state_id,
                    state_type=state_type,
                    parent=self.parent
                )
                
    def test_pseudo_state_parent_requirement(self):
        """Test that PseudoState requires a parent state."""
        with self.assertRaises(ValueError):
            PseudoState(
                state_id=self.state_id,
                state_type=self.state_type,
                parent=None
            )
            
    def test_pseudo_state_child_prevention(self):
        """Test that PseudoState cannot have child states."""
        parent = PseudoState(
            state_id="parent",
            state_type=self.state_type,
            parent=self.parent
        )
        
        with self.assertRaises(ValueError):
            State(
                state_id="child",
                state_type=StateType.SIMPLE,
                parent=parent
            )
            
    def test_pseudo_state_activation(self):
        """Test that PseudoState activation is transient."""
        state = PseudoState(
            state_id=self.state_id,
            state_type=self.state_type,
            parent=self.parent
        )
        
        state.activate()
        self.assertFalse(state.is_active)  # Pseudostates cannot remain active


class TestHistoryState(unittest.TestCase):
    """Test cases for the HistoryState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent = CompositeState(state_id="parent")
        self.region = self.parent.add_region("region1")
        self.state1 = State(state_id="state1", state_type=StateType.SIMPLE, parent=self.parent)
        self.state2 = State(state_id="state2", state_type=StateType.SIMPLE, parent=self.parent)
        
    def test_history_state_creation(self):
        """Test that a HistoryState can be created with valid parameters."""
        # Test shallow history
        shallow = HistoryState(
            state_id="shallow_history",
            state_type=StateType.SHALLOW_HISTORY,
            parent=self.parent
        )
        
        self.assertEqual(shallow.id, "shallow_history")
        self.assertEqual(shallow.type, StateType.SHALLOW_HISTORY)
        self.assertEqual(shallow.parent, self.parent)
        self.assertFalse(shallow.is_active)
        self.assertIsNone(shallow.last_active_state)
        
        # Test deep history
        deep = HistoryState(
            state_id="deep_history",
            state_type=StateType.DEEP_HISTORY,
            parent=self.parent
        )
        
        self.assertEqual(deep.id, "deep_history")
        self.assertEqual(deep.type, StateType.DEEP_HISTORY)
        self.assertEqual(deep.parent, self.parent)
        self.assertFalse(deep.is_active)
        self.assertIsNone(deep.last_active_state)
        
    def test_history_state_type_validation(self):
        """Test that HistoryState enforces valid state types."""
        # Invalid history state types
        invalid_types = [
            StateType.INITIAL,
            StateType.CHOICE,
            StateType.JUNCTION,
            StateType.ENTRY_POINT,
            StateType.EXIT_POINT,
            StateType.TERMINATE,
            StateType.SIMPLE,
            StateType.COMPOSITE,
            StateType.SUBMACHINE,
            StateType.FINAL
        ]
        
        for state_type in invalid_types:
            with self.assertRaises(ValueError):
                HistoryState(
                    state_id="invalid",
                    state_type=state_type,
                    parent=self.parent
                )
                
    def test_history_state_parent_requirement(self):
        """Test that HistoryState requires a composite parent state."""
        # No parent
        with self.assertRaises(ValueError):
            HistoryState(
                state_id="history",
                state_type=StateType.SHALLOW_HISTORY,
                parent=None
            )
            
        # Non-composite parent
        simple_parent = State(
            state_id="simple_parent",
            state_type=StateType.SIMPLE
        )
        
        with self.assertRaises(ValueError):
            HistoryState(
                state_id="history",
                state_type=StateType.SHALLOW_HISTORY,
                parent=simple_parent
            )
            
    def test_history_state_tracking(self):
        """Test history state tracking functionality."""
        history = HistoryState(
            state_id="history",
            state_type=StateType.SHALLOW_HISTORY,
            parent=self.parent
        )
        
        # Initially no history
        self.assertIsNone(history.last_active_state)
        
        # Record state activation
        self.state1.activate()
        history.record_active_state(self.state1)
        self.assertEqual(history.last_active_state, self.state1)
        
        # Record new state activation
        self.state1.deactivate()
        self.state2.activate()
        history.record_active_state(self.state2)
        self.assertEqual(history.last_active_state, self.state2)
        
        # Clear history
        history.clear_history()
        self.assertIsNone(history.last_active_state)
        
    def test_history_state_restoration(self):
        """Test history state restoration functionality."""
        history = HistoryState(
            state_id="history",
            state_type=StateType.SHALLOW_HISTORY,
            parent=self.parent
        )
        
        # Record and restore state
        self.state1.activate()
        history.record_active_state(self.state1)
        self.state1.deactivate()
        
        restored_state = history.get_restoration_state()
        self.assertEqual(restored_state, self.state1)
        
        # Test default transition when no history
        history.clear_history()
        self.assertIsNone(history.get_restoration_state())


class TestConnectionPointState(unittest.TestCase):
    """Test cases for the ConnectionPointState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent = CompositeState(state_id="parent")
        
    def test_connection_point_creation(self):
        """Test that connection points can be created with valid parameters."""
        # Test entry point
        entry_point = ConnectionPointState(
            state_id="entry",
            state_type=StateType.ENTRY_POINT,
            parent=self.parent
        )
        
        self.assertEqual(entry_point.id, "entry")
        self.assertEqual(entry_point.type, StateType.ENTRY_POINT)
        self.assertEqual(entry_point.parent, self.parent)
        
        # Test exit point
        exit_point = ConnectionPointState(
            state_id="exit",
            state_type=StateType.EXIT_POINT,
            parent=self.parent
        )
        
        self.assertEqual(exit_point.id, "exit")
        self.assertEqual(exit_point.type, StateType.EXIT_POINT)
        self.assertEqual(exit_point.parent, self.parent)
        
    def test_connection_point_validation(self):
        """Test connection point validation rules."""
        # Test invalid type
        with self.assertRaises(ValueError):
            ConnectionPointState(
                state_id="invalid",
                state_type=StateType.SIMPLE,
                parent=self.parent
            )
            
        # Test missing parent
        with self.assertRaises(ValueError):
            ConnectionPointState(
                state_id="no_parent",
                state_type=StateType.ENTRY_POINT,
                parent=None
            )
            
        # Test non-composite parent
        simple_parent = State(state_id="simple", state_type=StateType.SIMPLE)
        with self.assertRaises(ValueError):
            ConnectionPointState(
                state_id="invalid_parent",
                state_type=StateType.ENTRY_POINT,
                parent=simple_parent
            )
            
    def test_connection_point_transition_validation(self):
        """Test connection point transition validation."""
        entry_point = ConnectionPointState(
            state_id="entry",
            state_type=StateType.ENTRY_POINT,
            parent=self.parent
        )
        
        exit_point = ConnectionPointState(
            state_id="exit",
            state_type=StateType.EXIT_POINT,
            parent=self.parent
        )
        
        # Create mock transitions
        to_entry = Mock()
        to_entry.target = entry_point
        to_entry.source = None
        
        from_entry = Mock()
        from_entry.source = entry_point
        from_entry.target = None
        
        to_exit = Mock()
        to_exit.target = exit_point
        to_exit.source = None
        
        from_exit = Mock()
        from_exit.source = exit_point
        from_exit.target = None
        
        # Test entry point validation
        self.assertTrue(entry_point.validate_transition(to_entry))
        self.assertFalse(entry_point.validate_transition(from_entry))
        
        # Test exit point validation
        self.assertTrue(exit_point.validate_transition(from_exit))
        self.assertFalse(exit_point.validate_transition(to_exit))


class TestChoiceState(unittest.TestCase):
    """Test cases for the ChoiceState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent = CompositeState(state_id="parent")
        self.choice = ChoiceState(state_id="choice", parent=self.parent)
        self.target1 = State(state_id="target1", state_type=StateType.SIMPLE)
        self.target2 = State(state_id="target2", state_type=StateType.SIMPLE)
        
    def test_choice_state_creation(self):
        """Test that a ChoiceState can be created with valid parameters."""
        self.assertEqual(self.choice.id, "choice")
        self.assertEqual(self.choice.type, StateType.CHOICE)
        self.assertEqual(self.choice.parent, self.parent)
        self.assertIsNone(self.choice.default_transition)
        
    def test_choice_state_transitions(self):
        """Test choice state transition management."""
        # Create transitions with guards
        transition1 = Mock()
        transition1.source = self.choice
        transition1.target = self.target1
        transition1.evaluate_guard.return_value = False
        
        transition2 = Mock()
        transition2.source = self.choice
        transition2.target = self.target2
        transition2.evaluate_guard.return_value = True
        
        default = Mock()
        default.source = self.choice
        default.target = self.target1
        
        # Add transitions
        self.choice.add_outgoing_transition(transition1)
        self.choice.add_outgoing_transition(transition2)
        self.choice.set_default_transition(default)
        
        # Test transition selection
        selected = self.choice.select_transition({})
        self.assertEqual(selected, transition2)  # Second transition's guard is true
        
        # Test default selection when no guards are true
        transition2.evaluate_guard.return_value = False
        selected = self.choice.select_transition({})
        self.assertEqual(selected, default)
        
    def test_choice_state_validation(self):
        """Test choice state validation rules."""
        # Test invalid transition source
        invalid_transition = Mock()
        invalid_transition.source = self.target1  # Not the choice state
        
        with self.assertRaises(ValueError):
            self.choice.add_outgoing_transition(invalid_transition)
            
        with self.assertRaises(ValueError):
            self.choice.set_default_transition(invalid_transition)


class TestJunctionState(unittest.TestCase):
    """Test cases for the JunctionState class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent = CompositeState(state_id="parent")
        self.junction = JunctionState(state_id="junction", parent=self.parent)
        self.target1 = State(state_id="target1", state_type=StateType.SIMPLE)
        self.target2 = State(state_id="target2", state_type=StateType.SIMPLE)
        
    def test_junction_state_creation(self):
        """Test that a JunctionState can be created with valid parameters."""
        self.assertEqual(self.junction.id, "junction")
        self.assertEqual(self.junction.type, StateType.JUNCTION)
        self.assertEqual(self.junction.parent, self.parent)
        self.assertIsNone(self.junction.default_transition)
        
    def test_junction_state_transitions(self):
        """Test junction state transition management."""
        # Create transitions with static conditions
        transition1 = Mock()
        transition1.source = self.junction
        transition1.target = self.target1
        transition1.evaluate_guard.return_value = False
        
        transition2 = Mock()
        transition2.source = self.junction
        transition2.target = self.target2
        transition2.evaluate_guard.return_value = True
        
        default = Mock()
        default.source = self.junction
        default.target = self.target1
        
        # Add transitions
        self.junction.add_outgoing_transition(transition1)
        self.junction.add_outgoing_transition(transition2)
        self.junction.set_default_transition(default)
        
        # Test transition selection
        selected = self.junction.select_transition({})
        self.assertEqual(selected, transition2)  # Second transition's condition is true
        
        # Test default selection when no conditions are true
        transition2.evaluate_guard.return_value = False
        selected = self.junction.select_transition({})
        self.assertEqual(selected, default)
        
    def test_junction_state_validation(self):
        """Test junction state validation rules."""
        # Test invalid transition source
        invalid_transition = Mock()
        invalid_transition.source = self.target1  # Not the junction state
        
        with self.assertRaises(ValueError):
            self.junction.add_outgoing_transition(invalid_transition)
            
        with self.assertRaises(ValueError):
            self.junction.set_default_transition(invalid_transition)


if __name__ == '__main__':
    unittest.main() 