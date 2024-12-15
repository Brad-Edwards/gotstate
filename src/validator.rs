// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation module.
use crate::model::{ModelDefinition, StateDefinition, TransitionDefinition};

/// `Validator` acts as a high-level entry point to validate the entire model.
/// Implementation may delegate tasks to `StructuralValidator`, `RuntimeValidator`, and `RuleEngine`.
/// - `StructuralValidator`: Checks static model integrity (no missing states, invalid transitions).
/// - `RuntimeValidator`: Validates conditions at or before execution time.
/// - `RuleEngine`: Applies additional domain-specific validation rules.
///
/// This layering allows flexibility, but implementers should document which validators are used.

pub trait Validator: Send + Sync {
    type Model: ModelDefinition;
    type ValidationResult;
    type Error;

    fn validate_model(&self, model: &Self::Model) -> Result<Self::ValidationResult, Self::Error>;
}

/// `StructuralValidator` focuses on structural integrity checks.
///
/// **Key Considerations:**
/// - Checks for missing states, orphan transitions, or invalid event references.
pub trait StructuralValidator: Send + Sync {
    type Model: ModelDefinition;
    type Error;

    fn validate_model_structure(&self, model: &Self::Model) -> Result<(), Self::Error>;
}

/// `RuntimeValidator` checks runtime aspects of states and transitions.
///
/// **Key Considerations:**
/// - Ensures transitions are consistent and states can be activated safely at runtime.
pub trait RuntimeValidator: Send + Sync {
    type StateDef: StateDefinition;
    type TransitionDef: TransitionDefinition;
    type Error;

    fn validate_state_activation(&self, state: &Self::StateDef) -> Result<(), Self::Error>;
    fn validate_transition_consistency(
        &self,
        transition: &Self::TransitionDef,
    ) -> Result<(), Self::Error>;
}

/// `RuleEngine` applies domain-specific rules to the model.
///
/// **Key Considerations:**
/// - Allows extending validation with custom rules.
/// - `register_rule` lets users add their own rule checks.
pub trait RuleEngine: Send + Sync {
    type Model: ModelDefinition;
    type Error;

    fn apply_rules(&self, model: &Self::Model) -> Result<(), Self::Error>;

    fn register_rule(
        &mut self,
        rule_name: &str,
        rule_fn: Box<dyn Fn(&Self::Model) -> Result<(), Self::Error> + Send + Sync + 'static>,
    );
}

/// `RecoveryCoordinator` attempts to recover from validation or runtime errors.
///
/// **Key Considerations:**
/// - Allows automatic or semi-automatic healing after errors are detected.
pub trait RecoveryCoordinator: Send + Sync {
    type Model: ModelDefinition;
    type Error;

    fn recover_from_error(
        &self,
        model: &Self::Model,
        error: &Self::Error,
    ) -> Result<(), Self::Error>;
}
