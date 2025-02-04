import unittest
from unittest.mock import Mock, patch

from gotstate.core.event import Event, EventKind, EventPriority
from gotstate.core.machine.protocol_machine import ProtocolMachine
from gotstate.core.machine.basic_state_machine import BasicStateMachine
from gotstate.core.machine.machine_status import MachineStatus
from tests.unit.machine.machine_mocks import MockState


class TestProtocolMachine(unittest.TestCase):
    """Test cases for ProtocolMachine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.machine = ProtocolMachine("test_protocol")
        self.state1 = MockState("state1")
        self.state2 = MockState("state2")
        self.state3 = MockState("state3")

    def test_initial_state(self):
        """Test initial state of protocol machine.
        
        Verifies:
        1. Initial protocol name
        2. Empty rules and sequences
        3. No current state
        4. Proper lock initialization
        """
        self.assertEqual(self.machine.protocol_name, "test_protocol")
        self.assertEqual(len(self.machine.protocol_rules), 0)
        self.assertIsNone(self.machine._current_state)
        self.assertIsNotNone(self.machine._rule_lock)
        self.assertIsNotNone(self.machine._sequence_lock)

    def test_add_protocol_rule_success(self):
        """Test successful protocol rule addition.
        
        Verifies:
        1. Rule is added correctly
        2. Rule fields are preserved
        3. Rule is copied
        """
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "guard": lambda: True,
            "effect": lambda: None
        }
        
        self.machine.add_protocol_rule(rule)
        rules = self.machine.protocol_rules
        
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["operation"], "op1")
        self.assertEqual(rules[0]["source"], "state1")
        self.assertEqual(rules[0]["target"], "state2")
        self.assertIn("guard", rules[0])
        self.assertIn("effect", rules[0])

    def test_add_protocol_rule_validation(self):
        """Test protocol rule validation.
        
        Verifies:
        1. Invalid rule type rejected
        2. Missing required fields rejected
        3. Rule conflicts detected
        """
        # Test invalid rule type
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_protocol_rule([])
        self.assertIn("must be a dictionary", str(ctx.exception))

        # Test missing fields
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_protocol_rule({"operation": "op1"})
        self.assertIn("must contain fields", str(ctx.exception))

        # Test rule conflict
        rule1 = {
            "operation": "op1",
            "source": "state1",
            "target": "state2"
        }
        rule2 = {
            "operation": "op1",
            "source": "state1",
            "target": "state3"
        }
        
        self.machine.add_protocol_rule(rule1)
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_protocol_rule(rule2)
        self.assertIn("Rule conflict", str(ctx.exception))

    def test_add_sequence_rule_success(self):
        """Test successful sequence rule addition.
        
        Verifies:
        1. Rule is added correctly
        2. Rule is copied
        3. Event is tracked
        """
        rule = {"op1": ["op2", "op3"]}
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.machine.add_sequence_rule(rule)
            
            mock_track.assert_called_once()
            event_name, event_data = mock_track.call_args[0]
            self.assertEqual(event_name, "sequence_rule_added")
            self.assertEqual(event_data["rule"], rule)

    def test_add_sequence_rule_validation(self):
        """Test sequence rule validation.
        
        Verifies:
        1. Invalid rule type rejected
        """
        with self.assertRaises(ValueError) as ctx:
            self.machine.add_sequence_rule([])
        self.assertIn("must be a dictionary", str(ctx.exception))

    def test_validate_sequence(self):
        """Test operation sequence validation.
        
        Verifies:
        1. First operation always allowed
        2. Sequence rules enforced
        3. Multiple rules handled
        """
        # Add sequence rules
        rule1 = {"op1": ["op2", "op3"]}
        rule2 = {"op2": ["op4"]}
        self.machine.add_sequence_rule(rule1)
        self.machine.add_sequence_rule(rule2)

        # First operation should be allowed
        self.assertTrue(self.machine._validate_sequence("op1"))

        # Add operation to sequence
        self.machine._operation_sequence.append("op1")

        # Test allowed operations
        self.assertTrue(self.machine._validate_sequence("op2"))
        self.assertTrue(self.machine._validate_sequence("op3"))
        self.assertFalse(self.machine._validate_sequence("op4"))

        # Move to next operation
        self.machine._operation_sequence.append("op2")

        # Test next level of sequence
        self.assertTrue(self.machine._validate_sequence("op4"))
        self.assertFalse(self.machine._validate_sequence("op3"))

    def test_validate_operation(self):
        """Test operation validation.
        
        Verifies:
        1. State-based validation
        2. Sequence-based validation
        3. Event tracking on failure
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2"
        }
        self.machine.add_protocol_rule(rule)
        
        # No current state
        self.assertFalse(self.machine._validate_operation("op1"))
        
        # Set current state
        self.machine._current_state = self.state1
        
        # Valid operation
        self.assertTrue(self.machine._validate_operation("op1"))
        
        # Invalid operation
        with patch.object(self.machine, '_track_event') as mock_track:
            self.assertFalse(self.machine._validate_operation("op2"))
            mock_track.assert_called_with(
                "state_validation_failed",
                {"operation": "op2", "current_state": "state1"}
            )

    def test_apply_operation(self):
        """Test operation application.
        
        Verifies:
        1. State transition
        2. Guard evaluation
        3. Effect execution
        4. Sequence tracking
        5. Event tracking
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        
        guard = Mock(return_value=True)
        effect = Mock()
        
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "guard": guard,
            "effect": effect
        }
        self.machine.add_protocol_rule(rule)
        
        self.machine._current_state = self.state1
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.machine._apply_operation("op1")
            
            # Verify state change
            self.assertEqual(self.machine._current_state, self.state2)
            
            # Verify guard and effect called
            guard.assert_called_once()
            effect.assert_called_once()
            
            # Verify sequence updated
            self.assertEqual(self.machine._operation_sequence, ["op1"])
            
            # Verify event tracked
            mock_track.assert_called_with(
                "operation_applied",
                {"operation": "op1", "from_state": "state1", "to_state": "state2"}
            )

    def test_apply_operation_guard_failure(self):
        """Test operation application with guard failure.
        
        Verifies:
        1. Guard failure handling
        2. State preservation
        3. Event tracking
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        
        guard = Mock(return_value=False)
        
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "guard": guard
        }
        self.machine.add_protocol_rule(rule)
        
        self.machine._current_state = self.state1
        
        with patch.object(self.machine, '_track_event') as mock_track:
            with self.assertRaises(ValueError) as ctx:
                self.machine._apply_operation("op1")
            
            self.assertIn("Guard condition failed", str(ctx.exception))
            self.assertEqual(self.machine._current_state, self.state1)
            mock_track.assert_called_with(
                "guard_condition_failed",
                {"operation": "op1", "state": "state1"}
            )

    def test_process_event(self):
        """Test event processing.
        
        Verifies:
        1. Call event handling
        2. Operation validation
        3. Operation application
        4. Non-call event delegation
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2"
        }
        self.machine.add_protocol_rule(rule)
        
        # Initialize and activate the machine
        self.machine.initialize()
        self.machine._status = MachineStatus.ACTIVE
        self.machine._current_state = self.state1
        
        # Test call event
        event = Event("test", EventKind.CALL, EventPriority.NORMAL, data={"operation": "op1"})
        self.machine.process_event(event)
        self.assertTrue(event._consumed)  # Access internal field since property is read-only
        self.assertEqual(self.machine._current_state, self.state2)
        
        # Test non-call event
        event = Event("test", EventKind.SIGNAL, EventPriority.NORMAL)
        with patch.object(BasicStateMachine, 'process_event') as mock_process:
            self.machine.process_event(event)
            mock_process.assert_called_once_with(event)

    def test_process_event_validation(self):
        """Test event processing validation.
        
        Verifies:
        1. Missing operation handling
        2. Invalid operation handling
        3. Invalid sequence handling
        """
        # Test missing operation
        event = Event("test", EventKind.CALL, EventPriority.NORMAL, data={})
        with self.assertRaises(ValueError) as ctx:
            self.machine.process_event(event)
        self.assertIn("must specify operation", str(ctx.exception))
        
        # Test invalid operation
        event = Event("test", EventKind.CALL, EventPriority.NORMAL, data={"operation": "invalid"})
        with self.assertRaises(ValueError) as ctx:
            self.machine.process_event(event)
        self.assertIn("Invalid operation", str(ctx.exception))

    def test_initialize(self):
        """Test protocol machine initialization.
        
        Verifies:
        1. Initial state set
        2. State entry called
        3. Base initialization performed
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        
        with patch.object(BasicStateMachine, 'initialize') as mock_init:
            self.machine.initialize()
            
            mock_init.assert_called_once()
            self.assertEqual(self.machine._current_state, self.state1)
            self.state1.enter.assert_called_once()

    def test_terminate(self):
        """Test protocol machine termination.
        
        Verifies:
        1. Current state exit called
        2. Current state cleared
        3. Base termination performed
        """
        self.machine._current_state = self.state1
        
        with patch.object(BasicStateMachine, 'terminate') as mock_term:
            self.machine.terminate()
            
            self.state1.exit.assert_called_once()
            self.assertIsNone(self.machine._current_state)
            mock_term.assert_called_once()

    def test_clear_sequence(self):
        """Test sequence clearing.
        
        Verifies:
        1. Sequence cleared
        2. Event tracked
        """
        self.machine._operation_sequence = ["op1", "op2"]
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.machine.clear_sequence()
            
            self.assertEqual(len(self.machine._operation_sequence), 0)
            mock_track.assert_called_with("sequence_cleared", {})

    def test_validate_configuration_no_states(self):
        """Test configuration validation with no states.
        
        Verifies:
        1. Empty state machine validation fails
        """
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("must have at least one state", str(ctx.exception))

    def test_validate_configuration_invalid_source_state(self):
        """Test configuration validation with invalid source state.
        
        Verifies:
        1. Invalid source state detection
        """
        self.machine.add_state(self.state1)
        rule = {
            "operation": "op1",
            "source": "nonexistent",
            "target": "state1"
        }
        self.machine.add_protocol_rule(rule)
        
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Source state not found", str(ctx.exception))

    def test_validate_configuration_invalid_target_state(self):
        """Test configuration validation with invalid target state.
        
        Verifies:
        1. Invalid target state detection
        """
        self.machine.add_state(self.state1)
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "nonexistent"
        }
        self.machine.add_protocol_rule(rule)
        
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Target state not found", str(ctx.exception))

    def test_validate_configuration_invalid_guard(self):
        """Test configuration validation with invalid guard.
        
        Verifies:
        1. Non-callable guard detection
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "guard": "not_callable"
        }
        self.machine.add_protocol_rule(rule)
        
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Guard must be callable", str(ctx.exception))

    def test_validate_configuration_invalid_effect(self):
        """Test configuration validation with invalid effect.
        
        Verifies:
        1. Non-callable effect detection
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "effect": "not_callable"
        }
        self.machine.add_protocol_rule(rule)
        
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("Effect must be callable", str(ctx.exception))

    def test_validate_configuration_invalid_sequence_rule(self):
        """Test configuration validation with invalid sequence rule.
        
        Verifies:
        1. Invalid next operations type
        2. Invalid operation type in list
        """
        self.machine.add_state(self.state1)
        
        # Test non-list next operations
        self.machine.add_sequence_rule({"op1": "not_a_list"})
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("must be a list", str(ctx.exception))
        
        # Reset sequence rules
        self.machine._sequence_rules.clear()
        
        # Test non-string operation in list
        self.machine.add_sequence_rule({"op1": ["valid", 123]})
        with self.assertRaises(ValueError) as ctx:
            self.machine._validate_configuration()
        self.assertIn("must be strings", str(ctx.exception))

    def test_apply_operation_effect_error(self):
        """Test operation application with effect error.
        
        Verifies:
        1. Effect error handling
        2. Event tracking
        3. Error propagation
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        
        effect = Mock(side_effect=RuntimeError("Effect failed"))
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2",
            "effect": effect
        }
        self.machine.add_protocol_rule(rule)
        self.machine._current_state = self.state1
        
        with patch.object(self.machine, '_track_event') as mock_track:
            with self.assertRaises(RuntimeError):
                self.machine._apply_operation("op1")
            
            mock_track.assert_called_with(
                "operation_failed",
                {"operation": "op1", "error": "Effect failed"}
            )

    def test_process_event_no_operation(self):
        """Test event processing with missing operation.
        
        Verifies:
        1. Missing operation detection
        2. Error handling
        """
        event = Event("test", EventKind.CALL, EventPriority.NORMAL, data={})
        with self.assertRaises(ValueError) as ctx:
            self.machine.process_event(event)
        self.assertIn("must specify operation", str(ctx.exception))

    def test_process_event_sequence_error(self):
        """Test event processing with sequence error.
        
        Verifies:
        1. Sequence error detection in initial state
        2. Error handling
        """
        self.machine.add_state(self.state1)
        self.machine._current_state = self.state1
        
        event = Event("test", EventKind.CALL, EventPriority.NORMAL, data={"operation": "invalid"})
        with self.assertRaises(ValueError) as ctx:
            self.machine.process_event(event)
        self.assertIn("Invalid operation sequence", str(ctx.exception))

    def test_apply_operation_no_current_state(self):
        """Test operation application with no current state.
        
        Verifies:
        1. Error handling when no current state
        """
        with self.assertRaises(ValueError) as ctx:
            self.machine._apply_operation("op1")
        self.assertIn("No current state", str(ctx.exception))

    def test_apply_operation_no_matching_rule(self):
        """Test operation application with no matching rule.
        
        Verifies:
        1. Error handling when no rule matches
        """
        self.machine._current_state = self.state1
        
        with self.assertRaises(ValueError) as ctx:
            self.machine._apply_operation("op1")
        self.assertIn("No rule found for operation", str(ctx.exception))

    def test_apply_operation_target_state_missing(self):
        """Test operation application with missing target state.
        
        Verifies:
        1. Error handling when target state doesn't exist
        """
        self.machine._current_state = self.state1
        rule = {
            "operation": "op1",
            "source": "state1",
            "target": "state2"  # State2 not added to machine
        }
        self.machine.add_protocol_rule(rule)
        
        with self.assertRaises(ValueError) as ctx:
            self.machine._apply_operation("op1")
        self.assertIn("Target state not found", str(ctx.exception))

    def test_initialize_no_states(self):
        """Test initialization with no states.
        
        Verifies:
        1. Base initialization still performed
        2. No state change attempted
        """
        with patch.object(BasicStateMachine, 'initialize') as mock_init:
            self.machine.initialize()
            mock_init.assert_called_once()
            self.assertIsNone(self.machine._current_state)

    def test_process_event_invalid_operation_not_initial(self):
        """Test event processing with invalid operation in non-initial state.
        
        Verifies:
        1. Error handling for invalid operations
        2. Different error message when not in initial state
        """
        self.machine.add_state(self.state1)
        self.machine.add_state(self.state2)
        self.machine._current_state = self.state2  # Not the initial state
        
        event = Event("test", EventKind.CALL, EventPriority.NORMAL, data={"operation": "invalid"})
        with self.assertRaises(ValueError) as ctx:
            self.machine.process_event(event)
        self.assertIn("Invalid operation: invalid", str(ctx.exception))

    def test_validate_sequence_empty_rules(self):
        """Test sequence validation with empty rules.
        
        Verifies:
        1. First operation allowed with no rules
        2. Subsequent operations allowed with no rules
        """
        # First operation should be allowed
        self.assertTrue(self.machine._validate_sequence("op1"))
        
        # Add operation to sequence and test again
        self.machine._operation_sequence.append("op1")
        self.assertTrue(self.machine._validate_sequence("op2"))

    def test_validate_sequence_multiple_rules(self):
        """Test sequence validation with multiple overlapping rules.
        
        Verifies:
        1. Multiple rule handling
        2. Rule precedence
        """
        # Each rule defines its own valid next operations
        rule1 = {"op1": ["op2", "op3"]}
        rule2 = {"op1": ["op2", "op4"]}
        self.machine.add_sequence_rule(rule1)
        self.machine.add_sequence_rule(rule2)
        
        self.machine._operation_sequence.append("op1")
        
        # Only op2 is valid as it's the only operation allowed by both rules
        self.assertTrue(self.machine._validate_sequence("op2"))
        self.assertFalse(self.machine._validate_sequence("op3"))  # Only in rule1
        self.assertFalse(self.machine._validate_sequence("op4"))  # Only in rule2
        self.assertFalse(self.machine._validate_sequence("op5"))  # In neither rule

    def test_validate_operation_no_rules(self):
        """Test operation validation with no rules.
        
        Verifies:
        1. Operation validation without rules
        2. Event tracking
        """
        self.machine._current_state = self.state1
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.assertFalse(self.machine._validate_operation("op1"))
            mock_track.assert_called_with(
                "state_validation_failed",
                {"operation": "op1", "current_state": "state1"}
            )

    def test_validate_operation_sequence_tracking(self):
        """Test operation validation with sequence tracking.
        
        Verifies:
        1. Sequence validation tracking
        2. Event tracking for sequence failures
        """
        self.machine._current_state = self.state1
        rule = {"op1": ["op2"]}
        self.machine.add_sequence_rule(rule)
        
        # Add an operation that will make the next one invalid
        self.machine._operation_sequence.append("op1")
        
        with patch.object(self.machine, '_track_event') as mock_track:
            self.assertFalse(self.machine._validate_operation("op3"))
            mock_track.assert_called_with(
                "sequence_validation_failed",
                {"operation": "op3", "current_sequence": ["op1"]}
            )
