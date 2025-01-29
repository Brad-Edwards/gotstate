"""Unit tests for the Transition class and its subclasses.

Tests the transition management and execution functionality.
"""

import unittest
from unittest.mock import Mock, patch
from gotstate.core.state import State, StateType, PseudoState
from gotstate.core.event import Event, EventKind, EventPriority
from gotstate.core.transition import (
    Transition,
    TransitionKind,
    TransitionPriority,
    GuardCondition,
    TransitionEffect
)


class TestTransition(unittest.TestCase):
    """Test cases for the Transition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.source = State(state_id="source", state_type=StateType.SIMPLE)
        self.target = State(state_id="target", state_type=StateType.SIMPLE)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.SIGNAL,
            priority=EventPriority.NORMAL
        )
        self.guard = Mock(return_value=True)
        self.effect = Mock()
        
    def test_transition_creation(self):
        """Test that a Transition can be created with valid parameters."""
        transition = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            guard=self.guard,
            effect=self.effect,
            kind=TransitionKind.EXTERNAL,
            priority=TransitionPriority.NORMAL
        )
        
        self.assertEqual(transition.source, self.source)
        self.assertEqual(transition.target, self.target)
        self.assertEqual(transition.event, self.event)
        self.assertEqual(transition.guard, self.guard)
        self.assertEqual(transition.effect, self.effect)
        self.assertEqual(transition.kind, TransitionKind.EXTERNAL)
        self.assertEqual(transition.priority, TransitionPriority.NORMAL)
        
    def test_transition_validation(self):
        """Test transition validation rules."""
        # Test missing source
        with self.assertRaises(ValueError):
            Transition(
                source=None,
                target=self.target,
                event=self.event
            )
            
        # Test missing target for non-internal transition
        with self.assertRaises(ValueError):
            Transition(
                source=self.source,
                target=None,
                event=self.event,
                kind=TransitionKind.EXTERNAL
            )
            
        # Test invalid transition kind
        with self.assertRaises(ValueError):
            Transition(
                source=self.source,
                target=self.target,
                event=self.event,
                kind="invalid"
            )
            
    def test_transition_guard_evaluation(self):
        """Test transition guard condition evaluation."""
        transition = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            guard=self.guard
        )
        
        # Test guard evaluation with event data
        event_data = {"test": "data"}
        self.event._data = event_data
        
        self.assertTrue(transition.evaluate_guard())
        self.guard.assert_called_once_with(event_data)
        
        # Test guard evaluation when guard returns False
        self.guard.return_value = False
        self.assertFalse(transition.evaluate_guard())
        
    def test_transition_effect_execution(self):
        """Test transition effect execution."""
        transition = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            effect=self.effect
        )
        
        # Test effect execution with event data
        event_data = {"test": "data"}
        self.event._data = event_data
        
        transition.execute_effect()
        self.effect.assert_called_once_with(event_data)
        
    def test_transition_execution(self):
        """Test complete transition execution."""
        transition = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            guard=self.guard,
            effect=self.effect
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute transition
        result = transition.execute()
        
        self.assertTrue(result)
        self.guard.assert_called_once()
        self.effect.assert_called_once()
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_internal_transition(self):
        """Test internal transition behavior."""
        transition = Transition(
            source=self.source,
            target=None,
            event=self.event,
            effect=self.effect,
            kind=TransitionKind.INTERNAL
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute internal transition
        result = transition.execute()
        
        self.assertTrue(result)
        self.effect.assert_called_once()
        self.assertTrue(self.source.is_active)  # State remains active
        
    def test_transition_priority(self):
        """Test transition priority handling."""
        high_priority = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            priority=TransitionPriority.HIGH
        )
        
        normal_priority = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            priority=TransitionPriority.NORMAL
        )
        
        low_priority = Transition(
            source=self.source,
            target=self.target,
            event=self.event,
            priority=TransitionPriority.LOW
        )
        
        # Verify priority ordering
        self.assertTrue(high_priority.priority.value < normal_priority.priority.value)
        self.assertTrue(normal_priority.priority.value < low_priority.priority.value)
        
    def test_transition_to_pseudostate(self):
        """Test transitions to pseudostates."""
        pseudo_target = PseudoState(
            state_id="pseudo_target",
            state_type=StateType.CHOICE,
            parent=self.source
        )
        
        transition = Transition(
            source=self.source,
            target=pseudo_target,
            event=self.event
        )
        
        # Execute transition to pseudostate
        self.source.activate()
        result = transition.execute()
        
        self.assertTrue(result)
        self.assertFalse(self.source.is_active)
        self.assertFalse(pseudo_target.is_active)  # Pseudostates don't become active
        
    def test_completion_transition(self):
        """Test completion transition behavior."""
        transition = Transition(
            source=self.source,
            target=self.target,
            event=None,  # Completion transition has no triggering event
            kind=TransitionKind.EXTERNAL
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute completion transition
        result = transition.execute()
        
        self.assertTrue(result)
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)


class TestGuardCondition(unittest.TestCase):
    """Test cases for the GuardCondition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_data = {"value": 42}
        
    def test_guard_condition_creation(self):
        """Test that a GuardCondition can be created with valid parameters."""
        condition = lambda data: data["value"] > 0
        guard = GuardCondition(condition)
        
        self.assertTrue(guard.evaluate(self.event_data))
        
    def test_guard_condition_composition(self):
        """Test guard condition composition."""
        condition1 = GuardCondition(lambda data: data["value"] > 0)
        condition2 = GuardCondition(lambda data: data["value"] < 100)
        
        # Test AND composition
        composite = condition1 & condition2
        self.assertTrue(composite.evaluate(self.event_data))
        
        # Test OR composition
        composite = condition1 | GuardCondition(lambda data: data["value"] < 0)
        self.assertTrue(composite.evaluate(self.event_data))
        
        # Test NOT composition
        composite = ~GuardCondition(lambda data: data["value"] < 0)
        self.assertTrue(composite.evaluate(self.event_data))


class TestTransitionEffect(unittest.TestCase):
    """Test cases for the TransitionEffect class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_data = {"value": 42}
        
    def test_transition_effect_creation(self):
        """Test that a TransitionEffect can be created with valid parameters."""
        action = Mock()
        effect = TransitionEffect(action)
        
        effect.execute(self.event_data)
        action.assert_called_once_with(self.event_data)
        
    def test_transition_effect_composition(self):
        """Test transition effect composition."""
        action1 = Mock()
        action2 = Mock()
        
        effect1 = TransitionEffect(action1)
        effect2 = TransitionEffect(action2)
        
        # Test sequential composition
        composite = effect1 + effect2
        composite.execute(self.event_data)
        
        action1.assert_called_once_with(self.event_data)
        action2.assert_called_once_with(self.event_data)


if __name__ == '__main__':
    unittest.main() 