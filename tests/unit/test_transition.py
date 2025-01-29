"""Unit tests for the Transition class and its subclasses.

Tests the transition management and execution functionality.
"""

import unittest
from unittest.mock import Mock, patch
from gotstate.core.state import State, StateType, PseudoState, CompositeState, ChoiceState
from gotstate.core.event import Event, EventKind, EventPriority
from gotstate.core.transition import (
    Transition,
    TransitionKind,
    TransitionPriority,
    GuardCondition,
    TransitionEffect,
    ExternalTransition,
    InternalTransition,
    LocalTransition,
    CompoundTransition,
    ProtocolTransition,
    TimeTransition,
    ChangeTransition
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


class TestExternalTransition(unittest.TestCase):
    """Test cases for the ExternalTransition class."""
    
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
        
    def test_external_transition_execution(self):
        """Test external transition execution sequence."""
        transition = ExternalTransition(
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
        
        # Verify execution sequence
        self.assertTrue(result)
        self.guard.assert_called_once()
        self.effect.assert_called_once()
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_external_transition_composite_states(self):
        """Test external transition between composite states."""
        # Create composite states with regions
        source = CompositeState(state_id="source")
        source_region = source.add_region("region1")
        source_substate = State(state_id="source_sub", state_type=StateType.SIMPLE, parent=source)
        
        target = CompositeState(state_id="target")
        target_region = target.add_region("region1")
        target_substate = State(state_id="target_sub", state_type=StateType.SIMPLE, parent=target)
        
        transition = ExternalTransition(
            source=source,
            target=target,
            event=self.event
        )
        
        # Activate source state and substate
        source.activate()
        source_substate.activate()
        
        # Execute transition
        result = transition.execute()
        
        # Verify full state exit/entry
        self.assertTrue(result)
        self.assertFalse(source.is_active)
        self.assertFalse(source_substate.is_active)
        self.assertTrue(target.is_active)


class TestInternalTransition(unittest.TestCase):
    """Test cases for the InternalTransition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state = State(state_id="state", state_type=StateType.SIMPLE)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.SIGNAL,
            priority=EventPriority.NORMAL
        )
        self.effect = Mock()
        
    def test_internal_transition_execution(self):
        """Test internal transition execution."""
        transition = InternalTransition(
            source=self.state,
            event=self.event,
            effect=self.effect
        )
        
        # Activate state
        self.state.activate()
        
        # Execute transition
        result = transition.execute()
        
        # Verify state remains active and effect executed
        self.assertTrue(result)
        self.effect.assert_called_once()
        self.assertTrue(self.state.is_active)
        
    def test_internal_transition_validation(self):
        """Test internal transition validation rules."""
        # Test with different source and target
        with self.assertRaises(ValueError):
            InternalTransition(
                source=self.state,
                target=State(state_id="other", state_type=StateType.SIMPLE),
                event=self.event
            )


class TestLocalTransition(unittest.TestCase):
    """Test cases for the LocalTransition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parent = CompositeState(state_id="parent")
        self.source = State(state_id="source", state_type=StateType.SIMPLE, parent=self.parent)
        self.target = State(state_id="target", state_type=StateType.SIMPLE, parent=self.parent)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.SIGNAL,
            priority=EventPriority.NORMAL
        )
        self.effect = Mock()
        
    def test_local_transition_execution(self):
        """Test local transition execution."""
        transition = LocalTransition(
            source=self.source,
            target=self.target,
            event=self.event,
            effect=self.effect
        )
        
        # Activate states
        self.parent.activate()
        self.source.activate()
        
        # Execute transition
        result = transition.execute()
        
        # Verify minimal state changes
        self.assertTrue(result)
        self.effect.assert_called_once()
        self.assertTrue(self.parent.is_active)  # Parent remains active
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_local_transition_validation(self):
        """Test local transition validation rules."""
        # Test transition between states in different parents
        other_parent = CompositeState(state_id="other_parent")
        other_target = State(state_id="other_target", state_type=StateType.SIMPLE, parent=other_parent)
        
        with self.assertRaises(ValueError):
            LocalTransition(
                source=self.source,
                target=other_target,
                event=self.event
            )


class TestCompoundTransition(unittest.TestCase):
    """Test cases for the CompoundTransition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.source = State(state_id="source", state_type=StateType.SIMPLE)
        self.choice = ChoiceState(state_id="choice", parent=None)
        self.target = State(state_id="target", state_type=StateType.SIMPLE)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.SIGNAL,
            priority=EventPriority.NORMAL
        )
        
    def test_compound_transition_execution(self):
        """Test compound transition execution through pseudostates."""
        # Create transition segments
        segment1 = Transition(
            source=self.source,
            target=self.choice,
            event=self.event
        )
        
        segment2 = Transition(
            source=self.choice,
            target=self.target,
            guard=lambda data: True
        )
        
        transition = CompoundTransition(
            segments=[segment1, segment2],
            event=self.event
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute compound transition
        result = transition.execute()
        
        # Verify complete execution
        self.assertTrue(result)
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_compound_transition_validation(self):
        """Test compound transition validation rules."""
        # Test empty segments
        with self.assertRaises(ValueError):
            CompoundTransition(segments=[], event=self.event)
            
        # Test disconnected segments
        other_state = State(state_id="other", state_type=StateType.SIMPLE)
        segment1 = Transition(source=self.source, target=self.choice)
        segment2 = Transition(source=other_state, target=self.target)  # Not connected
        
        with self.assertRaises(ValueError):
            CompoundTransition(segments=[segment1, segment2], event=self.event)


class TestProtocolTransition(unittest.TestCase):
    """Test cases for the ProtocolTransition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.source = State(state_id="source", state_type=StateType.SIMPLE)
        self.target = State(state_id="target", state_type=StateType.SIMPLE)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.CALL,
            priority=EventPriority.NORMAL,
            data={"operation": "test_op", "args": [], "kwargs": {}}
        )
        
    def test_protocol_transition_execution(self):
        """Test protocol transition execution with operation validation."""
        # Create protocol transition with operation constraint
        transition = ProtocolTransition(
            source=self.source,
            target=self.target,
            event=self.event,
            operation="test_op"
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute transition
        result = transition.execute()
        
        # Verify execution and operation validation
        self.assertTrue(result)
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_protocol_transition_validation(self):
        """Test protocol transition validation rules."""
        transition = ProtocolTransition(
            source=self.source,
            target=self.target,
            event=self.event,
            operation="test_op"
        )
        
        # Test invalid operation
        self.event._data["operation"] = "invalid_op"
        result = transition.execute()
        self.assertFalse(result)
        
        # Test missing operation
        self.event._data.pop("operation")
        result = transition.execute()
        self.assertFalse(result)


class TestTimeTransition(unittest.TestCase):
    """Test cases for the TimeTransition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.source = State(state_id="source", state_type=StateType.SIMPLE)
        self.target = State(state_id="target", state_type=StateType.SIMPLE)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.TIME,
            priority=EventPriority.NORMAL,
            data={"type": "after", "time": 1000}  # 1 second
        )
        
    def test_time_transition_execution(self):
        """Test time transition execution."""
        transition = TimeTransition(
            source=self.source,
            target=self.target,
            event=self.event
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute transition
        result = transition.execute()
        
        # Verify execution
        self.assertTrue(result)
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_time_transition_validation(self):
        """Test time transition validation rules."""
        # Test invalid time event type
        invalid_event = Event(
            event_id="invalid",
            kind=EventKind.TIME,
            priority=EventPriority.NORMAL,
            data={"type": "invalid", "time": 1000}
        )
        
        with self.assertRaises(ValueError):
            TimeTransition(
                source=self.source,
                target=self.target,
                event=invalid_event
            )
            
        # Test negative time value
        invalid_event = Event(
            event_id="negative",
            kind=EventKind.TIME,
            priority=EventPriority.NORMAL,
            data={"type": "after", "time": -1000}
        )
        
        with self.assertRaises(ValueError):
            TimeTransition(
                source=self.source,
                target=self.target,
                event=invalid_event
            )


class TestChangeTransition(unittest.TestCase):
    """Test cases for the ChangeTransition class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.source = State(state_id="source", state_type=StateType.SIMPLE)
        self.target = State(state_id="target", state_type=StateType.SIMPLE)
        self.event = Event(
            event_id="test_event",
            kind=EventKind.CHANGE,
            priority=EventPriority.NORMAL,
            data={
                "condition": "value_changed",
                "target": "test_var",
                "old_value": 1,
                "new_value": 2
            }
        )
        
    def test_change_transition_execution(self):
        """Test change transition execution."""
        transition = ChangeTransition(
            source=self.source,
            target=self.target,
            event=self.event,
            condition=lambda old, new: new > old
        )
        
        # Activate source state
        self.source.activate()
        
        # Execute transition
        result = transition.execute()
        
        # Verify execution
        self.assertTrue(result)
        self.assertFalse(self.source.is_active)
        self.assertTrue(self.target.is_active)
        
    def test_change_transition_validation(self):
        """Test change transition validation rules."""
        transition = ChangeTransition(
            source=self.source,
            target=self.target,
            event=self.event,
            condition=lambda old, new: new > old
        )
        
        # Test condition not satisfied
        self.event._data["new_value"] = 0
        result = transition.execute()
        self.assertFalse(result)
        
        # Test missing change values
        self.event._data.pop("old_value")
        result = transition.execute()
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main() 