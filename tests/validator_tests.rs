// SPDX-License-Identifier: MIT OR Apache-2.0

use gotstate::validator::*;

mod mocks;
use mocks::*;

#[test]
fn test_validator_with_valid_model() {
    let validator = MockValidator;
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };

    let result = validator.validate_model(&model);
    assert!(result.is_ok());
}

#[test]
fn test_validator_with_invalid_model() {
    let validator = MockValidator;
    // Empty states
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![],
        events: vec![],
        transitions: vec![],
    };

    let result = validator.validate_model(&model);
    assert!(result.is_err());
}

#[test]
fn test_structural_validator_success() {
    let structural_validator = MockStructuralValidator::default();
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };

    assert!(structural_validator
        .validate_model_structure(&model)
        .is_ok());
}

#[test]
fn test_structural_validator_failure() {
    let structural_validator = MockStructuralValidator { should_fail: true };
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };

    let result = structural_validator.validate_model_structure(&model);
    assert!(result.is_err());
    assert_eq!(result.err().unwrap().0, "Structural validation failed");
}

#[test]
fn test_runtime_validator_state_activation() {
    let runtime_validator = MockRuntimeValidator::default();
    let state = MockState {
        name: "valid_state".to_string(),
        children: vec![],
    };

    assert!(runtime_validator.validate_state_activation(&state).is_ok());

    let runtime_validator = MockRuntimeValidator {
        invalid_states: vec!["bad_state".to_string()],
        invalid_transitions: vec![],
        inconsistent_transition: false,
    };
    let bad_state = MockState {
        name: "bad_state".to_string(),
        children: vec![],
    };
    let result = runtime_validator.validate_state_activation(&bad_state);
    assert!(result.is_err());
    assert_eq!(
        result.err().unwrap().0,
        "State bad_state cannot be activated"
    );
}

#[test]
fn test_runtime_validator_transition_consistency() {
    let runtime_validator = MockRuntimeValidator::default();
    let transition = MockTransition {
        from: MockState {
            name: "valid_from".to_string(),
            children: vec![],
        },
        to: MockState {
            name: "valid_to".to_string(),
            children: vec![],
        },
        event: MockEvent {
            id: 1,
            name: Some("test_event".to_string()),
            payload: None,
        },
        has_guard: false,
    };
    assert!(runtime_validator
        .validate_transition_consistency(&transition)
        .is_ok());

    let runtime_validator = MockRuntimeValidator {
        invalid_states: vec![],
        invalid_transitions: vec!["valid_from".to_string()],
        inconsistent_transition: false,
    };
    let result = runtime_validator.validate_transition_consistency(&transition);
    assert!(result.is_err());
    assert_eq!(
        result.err().unwrap().0,
        "Transition from valid_from is invalid"
    );
}

#[test]
fn test_runtime_validator_inconsistent_transition() {
    let runtime_validator = MockRuntimeValidator {
        invalid_states: vec![],
        invalid_transitions: vec![],
        inconsistent_transition: true,
    };
    let transition = MockTransition {
        from: MockState {
            name: "from".to_string(),
            children: vec![],
        },
        to: MockState {
            name: "to".to_string(),
            children: vec![],
        },
        event: MockEvent {
            id: 1,
            name: Some("event".to_string()),
            payload: None,
        },
        has_guard: false,
    };
    let result = runtime_validator.validate_transition_consistency(&transition);
    assert!(result.is_err());
    assert_eq!(result.err().unwrap().0, "Transition is inconsistent");
}

#[test]
fn test_rule_engine_success() {
    let rule_engine = MockRuleEngine::default();
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };
    assert!(rule_engine.apply_rules(&model).is_ok());
}

#[test]
fn test_rule_engine_failure() {
    let rule_engine = MockRuleEngine { fail_rules: true };
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };
    let result = rule_engine.apply_rules(&model);
    assert!(result.is_err());
    assert_eq!(result.err().unwrap().0, "Rule engine failed a rule");
}

#[test]
fn test_recovery_coordinator_success() {
    let recovery = MockRecoveryCoordinator::default();
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };
    assert!(recovery
        .recover_from_error(&model, &MockError("some error".to_string()))
        .is_ok());
}

#[test]
fn test_recovery_coordinator_failure() {
    let recovery = MockRecoveryCoordinator { can_recover: false };
    let model = MockModel {
        root: MockState {
            name: "root".to_string(),
            children: vec![],
        },
        states: vec![MockState {
            name: "root".to_string(),
            children: vec![],
        }],
        events: vec![],
        transitions: vec![],
    };
    let result = recovery.recover_from_error(&model, &MockError("fatal".to_string()));
    assert!(result.is_err());
    assert_eq!(result.err().unwrap().0, "Cannot recover from: fatal");
}
