# hsm/core/tests/test_validation.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
"""Unit tests for hsm.core.validation module."""
from unittest.mock import Mock

import pytest

from hsm.core.errors import ValidationError
from hsm.core.validation import ValidationContext, ValidationResult, ValidationRule, ValidationSeverity, Validator
from hsm.interfaces.abc import AbstractAction, AbstractEvent, AbstractGuard, AbstractState, AbstractTransition
from hsm.interfaces.types import StateID


class TestValidationSeverity:
    """Tests for ValidationSeverity enum."""

    def test_enum_values(self):
        """Verify expected enum values."""
        assert ValidationSeverity.ERROR.name == "ERROR"
        assert ValidationSeverity.WARNING.name == "WARNING"
        assert ValidationSeverity.INFO.name == "INFO"


class TestValidationContext:
    """Tests for ValidationContext class."""

    @pytest.fixture
    def mock_state(self):
        """Fixture for a mock state."""
        state = Mock(spec=AbstractState)
        state.get_id.return_value = "S1"
        return state

    @pytest.fixture
    def mock_transition(self, mock_state):
        """Fixture for a mock transition."""
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state
        return transition

    def test_initialization(self, mock_state, mock_transition):
        """Test initialization with states and transitions."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        assert context.states == [mock_state]
        assert context.transitions == [mock_transition]
        assert context.initial_state == mock_state
        assert context.current_results == []

    def test_add_result(self, mock_state, mock_transition):
        """Test adding validation results."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        context.add_result("ERROR", "Test error")
        assert len(context.current_results) == 1
        assert context.current_results[0].severity == "ERROR"
        assert context.current_results[0].message == "Test error"
        assert context.current_results[0].context == {}

    def test_add_result_with_context(self, mock_state, mock_transition):
        """Test adding results with context data."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        context.add_result("WARNING", "Test warning", {"key": "value"})
        assert context.current_results[0].context == {"key": "value"}

    def test_get_state_by_id(self, mock_state, mock_transition):
        """Test state lookup by ID."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        found_state = context.get_state_by_id("S1")
        assert found_state == mock_state

    def test_get_state_by_id_not_found(self, mock_state, mock_transition):
        """Test lookup for nonexistent state."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        not_found_state = context.get_state_by_id("S2")
        assert not_found_state is None

    def test_state_exists(self, mock_state, mock_transition):
        """Test state existence check."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        assert context.state_exists("S1")
        assert not context.state_exists("S2")


class TestValidationRule:
    """Tests for ValidationRule class."""

    def test_rule_creation(self):
        """Test creating validation rules."""
        mock_check = Mock()
        rule = ValidationRule("test_rule", mock_check, ValidationSeverity.ERROR, "Test description")
        assert rule.name == "test_rule"
        assert rule.check == mock_check
        assert rule.severity == ValidationSeverity.ERROR
        assert rule.description == "Test description"

    def test_rule_immutability(self):
        """Verify rules are immutable."""
        rule = ValidationRule("test_rule", Mock(), ValidationSeverity.INFO, "Test")
        with pytest.raises(Exception):
            rule.name = "new_name"
        with pytest.raises(Exception):
            rule.severity = ValidationSeverity.ERROR


class TestValidator:
    """Tests for Validator class."""

    @pytest.fixture
    def mock_state(self):
        """Fixture for a mock state."""
        state = Mock(spec=AbstractState)
        state.get_id.return_value = "S1"
        return state

    @pytest.fixture
    def mock_transition(self, mock_state):
        """Fixture for a mock transition."""
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state
        transition.get_actions.return_value = []  # Return empty list of actions by default
        transition.get_guard.return_value = None  # Return no guard by default
        return transition

    def test_initialization(self, mock_state, mock_transition):
        """Test validator initialization."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        assert validator._context.states == [mock_state]
        assert validator._context.transitions == [mock_transition]
        assert validator._context.initial_state == mock_state
        assert len(validator._structure_rules) > 0  # Default rules added
        assert len(validator._behavior_rules) > 0
        assert len(validator._data_rules) > 0

    def test_initialization_no_states(self, mock_state, mock_transition):
        """Test initialization with no states."""
        with pytest.raises(ValidationError, match="State machine must have at least one state"):
            Validator([], [mock_transition], mock_state)

    def test_initialization_invalid_initial_state(self, mock_state, mock_transition):
        """Test initialization with invalid initial state."""
        other_state = Mock(spec=AbstractState)
        with pytest.raises(ValidationError, match="Initial state must be in states list"):
            Validator([mock_state], [mock_transition], other_state)

    def test_add_rule(self, mock_state, mock_transition):
        """Test adding custom rules."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        mock_check = Mock(return_value=True)
        validator.add_rule("custom_rule", mock_check, ValidationSeverity.WARNING, "Custom rule", "structure")
        assert "custom_rule" in validator._structure_rules

    def test_add_rule_invalid_type(self, mock_state, mock_transition):
        """Test adding a rule with an invalid type."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        with pytest.raises(ValidationError, match="Invalid rule type: invalid_type"):
            validator.add_rule("custom_rule", Mock(), ValidationSeverity.INFO, "Custom", "invalid_type")

    def test_add_rule_duplicate_name(self, mock_state, mock_transition):
        """Test adding a rule with a duplicate name."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule("custom_rule", lambda ctx: True, ValidationSeverity.INFO, "First rule", "structure")

        with pytest.raises(ValidationError, match="Duplicate rule name: custom_rule"):
            validator.add_rule("custom_rule", lambda ctx: True, ValidationSeverity.WARNING, "Second rule", "structure")

    def test_validate_structure_success(self, mock_state, mock_transition):
        """Test successful structural validation."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        results = validator.validate_structure()
        assert len(results) == 0

    def test_validate_structure_failure(self, mock_state, mock_transition):
        """Test structural validation failure."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule("failing_rule", lambda ctx: False, ValidationSeverity.ERROR, "Fails", "structure")
        results = validator.validate_structure()
        assert len(results) == 1
        assert results[0].severity == "ERROR"

    def test_validate_structure_exception(self, mock_state, mock_transition):
        """Test handling exceptions during structural validation."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule(
            "error_rule", Mock(side_effect=Exception("Test error")), ValidationSeverity.ERROR, "Errors", "structure"
        )
        results = validator.validate_structure()
        assert len(results) == 1
        assert results[0].severity == "ERROR"
        assert "Test error" in results[0].message

    def test_validate_behavior_success(self, mock_state, mock_transition):
        """Test successful behavioral validation."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        results = validator.validate_behavior()
        assert len(results) == 0

    def test_validate_behavior_failure(self, mock_state, mock_transition):
        """Test behavioral validation failure."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule("failing_behavior_rule", lambda ctx: False, ValidationSeverity.WARNING, "Fails", "behavior")
        results = validator.validate_behavior()
        assert len(results) == 1
        assert results[0].severity == "WARNING"

    def test_validate_behavior_exception(self, mock_state, mock_transition):
        """Test handling exceptions during behavioral validation."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule(
            "error_behavior_rule",
            Mock(side_effect=Exception("Behavior error")),
            ValidationSeverity.WARNING,
            "Errors",
            "behavior",
        )
        results = validator.validate_behavior()
        assert len(results) == 1
        assert results[0].severity == "ERROR"
        assert "Behavior error" in results[0].message

    def test_validate_data_success(self, mock_state, mock_transition):
        """Test successful data validation."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        results = validator.validate_data()
        assert len(results) == 0

    def test_validate_data_failure(self, mock_state, mock_transition):
        """Test data validation failure."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule("failing_data_rule", lambda ctx: False, ValidationSeverity.ERROR, "Fails", "data")
        results = validator.validate_data()
        assert len(results) == 1
        assert results[0].severity == "ERROR"

    def test_validate_data_exception(self, mock_state, mock_transition):
        """Test handling exceptions during data validation."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        validator.add_rule(
            "error_data_rule", Mock(side_effect=Exception("Data error")), ValidationSeverity.ERROR, "Errors", "data"
        )
        results = validator.validate_data()
        assert len(results) == 1
        assert results[0].severity == "ERROR"
        assert "Data error" in results[0].message

    def test_check_no_orphan_states_all_reachable(self):
        """Test no_orphan_states with all states reachable."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S2"))
        t1 = Mock(
            spec=AbstractTransition,
            get_source_state=Mock(return_value=s1),
            get_target_state=Mock(return_value=s2),
        )
        context = ValidationContext([s1, s2], [t1], s1)
        validator = Validator([s1, s2], [t1], s1)
        assert validator._check_no_orphan_states(context)

    def test_check_no_orphan_states_unreachable(self):
        """Test no_orphan_states with unreachable states."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S2"))
        t1 = Mock(
            spec=AbstractTransition,
            get_source_state=Mock(return_value=s1),
            get_target_state=Mock(return_value=s1),
        )
        context = ValidationContext([s1, s2], [t1], s1)
        validator = Validator([s1, s2], [t1], s1)
        assert not validator._check_no_orphan_states(context)
        assert context.current_results[0].severity == "ERROR"
        assert "unreachable_states" in context.current_results[0].context

    def test_check_valid_transitions(self):
        """Test valid_transitions with valid transitions."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        t1 = Mock(
            spec=AbstractTransition,
            get_source_state=Mock(return_value=s1),
            get_target_state=Mock(return_value=s1),
        )
        context = ValidationContext([s1], [t1], s1)
        validator = Validator([s1], [t1], s1)
        assert validator._check_valid_transitions(context)

    def test_check_valid_transitions_invalid_source(self):
        """Test valid_transitions with invalid source state."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S2"))
        t1 = Mock(
            spec=AbstractTransition,
            get_source_state=Mock(return_value=s2),
            get_target_state=Mock(return_value=s1),
        )
        context = ValidationContext([s1], [t1], s1)
        validator = Validator([s1], [t1], s1)
        assert not validator._check_valid_transitions(context)
        assert context.current_results[0].severity == "ERROR"
        assert "nonexistent source state" in context.current_results[0].message

    def test_check_valid_transitions_invalid_target(self):
        """Test valid_transitions with invalid target state."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S2"))
        t1 = Mock(
            spec=AbstractTransition,
            get_source_state=Mock(return_value=s1),
            get_target_state=Mock(return_value=s2),
        )
        context = ValidationContext([s1], [t1], s1)
        validator = Validator([s1], [t1], s1)
        assert not validator._check_valid_transitions(context)
        assert context.current_results[0].severity == "ERROR"
        assert "nonexistent target state" in context.current_results[0].message

    def test_check_unique_state_ids(self):
        """Test unique_state_ids with unique IDs."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S2"))
        context = ValidationContext([s1, s2], [], s1)
        validator = Validator([s1, s2], [], s1)
        assert validator._check_unique_state_ids(context)

    def test_check_unique_state_ids_duplicate(self):
        """Test unique_state_ids with duplicate IDs."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        context = ValidationContext([s1, s2], [], s1)
        validator = Validator([s1, s2], [], s1)
        assert not validator._check_unique_state_ids(context)
        assert context.current_results[0].severity == "ERROR"
        assert "Duplicate state ID found" in context.current_results[0].message

    def test_add_rule_invalid_severity(self, mock_state, mock_transition):
        """Test adding a rule with invalid severity."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        with pytest.raises(TypeError):
            validator.add_rule("custom_rule", Mock(), "INVALID", "Custom rule", "structure")

    def test_check_guard_safety_with_invalid_guard(self, mock_state):
        """Test guard safety check with invalid guard."""
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state
        transition.get_guard.return_value = Mock()  # Non-AbstractGuard object

        context = ValidationContext([mock_state], [transition], mock_state)
        validator = Validator([mock_state], [transition], mock_state)

        assert not validator._check_guard_safety(context)
        assert context.current_results[0].severity == "WARNING"
        assert "does not implement AbstractGuard" in context.current_results[0].message

    def test_check_action_safety_with_invalid_action(self, mock_state):
        """Test action safety check with invalid action."""
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state
        transition.get_actions.return_value = [Mock()]  # Non-AbstractAction object

        context = ValidationContext([mock_state], [transition], mock_state)
        validator = Validator([mock_state], [transition], mock_state)

        assert not validator._check_action_safety(context)
        assert context.current_results[0].severity == "WARNING"
        assert "does not implement AbstractAction" in context.current_results[0].message

    def test_check_data_isolation_with_shared_data(self, mock_state):
        """Test data isolation check with shared data."""
        validator = Validator([mock_state], [], mock_state)

        # Mock the internal method to simulate shared data
        validator._state_shares_data_unintentionally = Mock(return_value=True)

        context = ValidationContext([mock_state], [], mock_state)
        assert not validator._check_data_isolation(context)
        assert context.current_results[0].severity == "ERROR"
        assert "shares data outside of isolation rules" in context.current_results[0].message

    def test_validate_with_multiple_rules(self, mock_state, mock_transition):
        """Test validation with multiple custom rules."""
        validator = Validator([mock_state], [mock_transition], mock_state)

        # Add multiple rules with different severities
        validator.add_rule("rule1", lambda ctx: True, ValidationSeverity.INFO, "Info rule", "structure")
        validator.add_rule("rule2", lambda ctx: False, ValidationSeverity.WARNING, "Warning rule", "structure")

        results = validator.validate_structure()
        assert len(results) == 1
        assert results[0].severity == "WARNING"
        assert "Warning rule" in results[0].message

    def test_get_state_by_id_edge_cases(self):
        """Test edge cases for state ID lookup."""
        s1 = Mock(spec=AbstractState, get_id=Mock(return_value="S1"))
        s2 = Mock(spec=AbstractState, get_id=Mock(return_value="S2"))
        context = ValidationContext([s1, s2], [], s1)

        assert context.get_state_by_id("S1") == s1
        assert context.get_state_by_id("S2") == s2
        assert context.get_state_by_id("NONEXISTENT") is None
        assert context.get_state_by_id("") is None

    def test_add_rule_invalid_rule_type(self, mock_state, mock_transition):
        """Test adding a rule with invalid rule type."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        with pytest.raises(ValidationError, match="Invalid rule type: invalid_type"):
            validator.add_rule("custom_rule", lambda ctx: True, ValidationSeverity.INFO, "Test rule", "invalid_type")

    def test_check_guard_safety_with_none_guard(self, mock_state):
        """Test guard safety check with None guard."""
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state
        transition.get_guard.return_value = None

        context = ValidationContext([mock_state], [transition], mock_state)
        validator = Validator([mock_state], [transition], mock_state)

        assert validator._check_guard_safety(context)
        assert len(context.current_results) == 0

    def test_check_action_safety_with_empty_actions(self, mock_state):
        """Test action safety check with empty actions list."""
        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state
        transition.get_actions.return_value = []

        context = ValidationContext([mock_state], [transition], mock_state)
        validator = Validator([mock_state], [transition], mock_state)

        assert validator._check_action_safety(context)
        assert len(context.current_results) == 0

    def test_check_data_isolation_no_states(self):
        """Test data isolation check with no states."""
        # Create context with no states
        context = ValidationContext([], [], None)

        # Create validator with a single state that's used as both the state and initial state
        mock_state = Mock(spec=AbstractState)
        validator = Validator([mock_state], [], mock_state)

        assert validator._check_data_isolation(context)
        assert len(context.current_results) == 0

    def test_validation_context_add_result_with_empty_context(self):
        """Test adding validation result with empty context dict."""
        context = ValidationContext([], [], None)
        context.add_result(ValidationSeverity.INFO.name, "Test message")

        assert len(context.current_results) == 1
        assert context.current_results[0].context == {}

    def test_validation_context_state_exists_empty_states(self):
        """Test state existence check with empty states list."""
        context = ValidationContext([], [], None)
        assert not context.state_exists("any_id")

    def test_validation_context_get_state_by_id_empty_states(self):
        """Test getting state by ID with empty states list."""
        context = ValidationContext([], [], None)
        assert context.get_state_by_id("any_id") is None

    def test_validate_behavior_with_no_rules(self, mock_state, mock_transition):
        """Test behavior validation with no custom rules."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        # Clear default behavior rules
        validator._behavior_rules = {}

        results = validator.validate_behavior()
        assert len(results) == 0

    def test_validate_data_with_no_rules(self, mock_state, mock_transition):
        """Test data validation with no custom rules."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        # Clear default data rules
        validator._data_rules = {}

        results = validator.validate_data()
        assert len(results) == 0

    def test_validate_structure_with_no_rules(self, mock_state, mock_transition):
        """Test structure validation with no custom rules."""
        validator = Validator([mock_state], [mock_transition], mock_state)
        # Clear default structure rules
        validator._structure_rules = {}

        results = validator.validate_structure()
        assert len(results) == 0

    def test_validation_context_add_result_with_none_context(self):
        """Test adding validation result with None context."""
        context = ValidationContext([], [], None)
        context.add_result(ValidationSeverity.WARNING.name, "Test message", None)

        assert len(context.current_results) == 1
        assert context.current_results[0].context == {}

    def test_check_no_orphan_states_no_transitions(self, mock_state):
        """Test orphan states check with no transitions."""
        context = ValidationContext([mock_state], [], mock_state)
        validator = Validator([mock_state], [], mock_state)

        assert not validator._check_no_orphan_states(context)
        assert len(context.current_results) == 1
        assert context.current_results[0].severity == "ERROR"
        assert "must have at least one transition" in context.current_results[0].message

    def test_check_valid_transitions_with_multiple_transitions(self, mock_state):
        """Test transition validation with multiple transitions."""
        state1 = Mock(spec=AbstractState)
        state1.get_id.return_value = "S1"
        state2 = Mock(spec=AbstractState)
        state2.get_id.return_value = "S2"

        transition1 = Mock(spec=AbstractTransition)
        transition1.get_source_state.return_value = state1
        transition1.get_target_state.return_value = state2

        transition2 = Mock(spec=AbstractTransition)
        transition2.get_source_state.return_value = state2
        transition2.get_target_state.return_value = state1

        context = ValidationContext([state1, state2], [transition1, transition2], state1)
        validator = Validator([state1, state2], [transition1, transition2], state1)

        assert validator._check_valid_transitions(context)
        assert len(context.current_results) == 0

    def test_validation_severity_comparison(self):
        """Test ValidationSeverity enum comparison."""
        assert ValidationSeverity.ERROR.value > ValidationSeverity.WARNING.value
        assert ValidationSeverity.WARNING.value > ValidationSeverity.INFO.value

    def test_validation_context_initialization_empty_lists(self):
        """Test ValidationContext initialization with empty lists."""
        context = ValidationContext([], [], None)
        assert context.states == []
        assert context.transitions == []
        assert context.initial_state is None
        assert context._state_ids == set()

    def test_validation_context_initialization_with_data(self, mock_state, mock_transition):
        """Test ValidationContext initialization with actual data."""
        context = ValidationContext([mock_state], [mock_transition], mock_state)
        assert context.states == [mock_state]
        assert context.transitions == [mock_transition]
        assert context.initial_state == mock_state
        assert context._state_ids == {mock_state.get_id()}

    def test_check_valid_transitions_with_invalid_source_state(self, mock_state):
        """Test transition validation with invalid source state."""
        invalid_state = Mock(spec=AbstractState)
        invalid_state.get_id.return_value = "INVALID"

        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = invalid_state
        transition.get_target_state.return_value = mock_state

        context = ValidationContext([mock_state], [transition], mock_state)
        validator = Validator([mock_state], [transition], mock_state)

        assert not validator._check_valid_transitions(context)
        assert len(context.current_results) == 1
        assert "nonexistent source state" in context.current_results[0].message

    def test_check_valid_transitions_with_invalid_target_state(self, mock_state):
        """Test transition validation with invalid target state."""
        invalid_state = Mock(spec=AbstractState)
        invalid_state.get_id.return_value = "INVALID"

        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = invalid_state

        context = ValidationContext([mock_state], [transition], mock_state)
        validator = Validator([mock_state], [transition], mock_state)

        assert not validator._check_valid_transitions(context)
        assert len(context.current_results) == 1
        assert "nonexistent target state" in context.current_results[0].message

    def test_check_no_orphan_states_with_unreachable_state(self, mock_state):
        """Test orphan states check with unreachable state."""
        state2 = Mock(spec=AbstractState)
        state2.get_id.return_value = "S2"

        transition = Mock(spec=AbstractTransition)
        transition.get_source_state.return_value = mock_state
        transition.get_target_state.return_value = mock_state

        context = ValidationContext([mock_state, state2], [transition], mock_state)
        validator = Validator([mock_state, state2], [transition], mock_state)

        assert not validator._check_no_orphan_states(context)
        assert len(context.current_results) == 1
        assert "not reachable from initial state" in context.current_results[0].message

    def test_validation_rule_immutability(self):
        """Test that ValidationRule instances are immutable."""
        rule = ValidationRule("test", lambda ctx: True, ValidationSeverity.INFO, "Test rule")

        with pytest.raises(Exception):
            rule.name = "new_name"
        with pytest.raises(Exception):
            rule.check = lambda ctx: False
        with pytest.raises(Exception):
            rule.severity = ValidationSeverity.ERROR
        with pytest.raises(Exception):
            rule.description = "New description"

    def test_validation_severity_str_representation(self):
        """Test string representation of ValidationSeverity enum."""
        assert str(ValidationSeverity.ERROR) == "ValidationSeverity.ERROR"
        assert str(ValidationSeverity.WARNING) == "ValidationSeverity.WARNING"
        assert str(ValidationSeverity.INFO) == "ValidationSeverity.INFO"

    def test_validation_result_equality(self):
        """Test equality comparison of ValidationResult objects."""
        result1 = ValidationResult("ERROR", "Test message", {"key": "value"})
        result2 = ValidationResult("ERROR", "Test message", {"key": "value"})
        result3 = ValidationResult("WARNING", "Test message", {"key": "value"})

        assert result1 == result2
        assert result1 != result3
        assert result1 != "not a validation result"

    def test_validation_result_repr(self):
        """Test string representation of ValidationResult."""
        result = ValidationResult("ERROR", "Test message", {"key": "value"})
        expected = "ValidationResult(severity='ERROR', message='Test message', context={'key': 'value'})"
        assert repr(result) == expected

    def test_validation_context_add_multiple_results(self):
        """Test adding multiple validation results to context."""
        context = ValidationContext([], [], None)

        context.add_result("ERROR", "First error")
        context.add_result("WARNING", "First warning")
        context.add_result("INFO", "First info")

        assert len(context.current_results) == 3
        assert [r.severity for r in context.current_results] == ["ERROR", "WARNING", "INFO"]
        assert [r.message for r in context.current_results] == ["First error", "First warning", "First info"]
