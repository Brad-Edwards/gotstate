// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validator module.
use crate::model::{ModelDefinition, StateDefinition, TransitionDefinition};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Extension of ModelDefinition to include associated types for State and Transition
pub trait ModelTypes {
    type StateDef: StateDefinition;
    type TransitionDef: TransitionDefinition;
}

/// `Validator` acts as a high-level entry point to validate the entire model.
/// - `RuntimeValidator`: Validates conditions at or before execution time.
/// - `RuleEngine`: Applies additional domain-specific validation rules.
///
/// This layering allows flexibility, but implementers should document which validators are used.
pub trait Validator: Send + Sync {
    type Model: ModelDefinition + ModelTypes;
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

/// An enumeration of all possible validation errors encountered during the validation process.
#[derive(Debug)]
pub enum ValidationError {
    StructuralError(String),
    RuntimeError(String),
    RuleError(String),
    RecoveryError(String),
    UnknownError(String),
}

/// A summary of the validation process, indicating whether the model is valid and listing any errors.
#[derive(Debug)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
}

/// A validator that combines structural, runtime, rule-based, and recovery validation.
pub struct FsmValidator<M, S, R, RL, RC>
where
    M: ModelDefinition + ModelTypes,
    S: StructuralValidator<Model = M, Error = ValidationError>,
    R: RuntimeValidator<
        Error = ValidationError,
        StateDef = <M as ModelTypes>::StateDef,
        TransitionDef = <M as ModelTypes>::TransitionDef,
    >,
    RL: RuleEngine<Model = M, Error = ValidationError>,
    RC: RecoveryCoordinator<Model = M, Error = ValidationError>,
{
    structural_validator: Arc<S>,
    runtime_validator: Arc<R>,
    rule_engine: Arc<Mutex<RL>>,
    recovery_coordinator: Option<Arc<RC>>,
}

impl<M, S, R, RL, RC> FsmValidator<M, S, R, RL, RC>
where
    M: ModelDefinition + ModelTypes,
    S: StructuralValidator<Model = M, Error = ValidationError>,
    R: RuntimeValidator<
        Error = ValidationError,
        StateDef = M::StateDef,
        TransitionDef = M::TransitionDef,
    >,
    RL: RuleEngine<Model = M, Error = ValidationError>,
    RC: RecoveryCoordinator<Model = M, Error = ValidationError>,
{
    /// Creates a new `FsmValidator` instance from the given structural, runtime, and rule validators,
    /// and an optional recovery coordinator.
    pub fn new(
        structural_validator: S,
        runtime_validator: R,
        rule_engine: RL,
        recovery_coordinator: Option<RC>,
    ) -> Self {
        Self {
            structural_validator: Arc::new(structural_validator),
            runtime_validator: Arc::new(runtime_validator),
            rule_engine: Arc::new(Mutex::new(rule_engine)),
            recovery_coordinator: recovery_coordinator.map(Arc::new),
        }
    }
}

impl<M, S, R, RL, RC> Validator for FsmValidator<M, S, R, RL, RC>
where
    M: ModelDefinition
        + ModelTypes<
            StateDef = <M as ModelDefinition>::StateDefinition,
            TransitionDef = <M as ModelDefinition>::TransitionDefinition,
        >,
    S: StructuralValidator<Model = M, Error = ValidationError>,
    R: RuntimeValidator<
        Error = ValidationError,
        StateDef = M::StateDef,
        TransitionDef = M::TransitionDef,
    >,
    RL: RuleEngine<Model = M, Error = ValidationError>,
    RC: RecoveryCoordinator<Model = M, Error = ValidationError>,
{
    type Model = M;
    type ValidationResult = ValidationResult;
    type Error = ValidationError;

    fn validate_model(&self, model: &Self::Model) -> Result<Self::ValidationResult, Self::Error> {
        let mut errors = Vec::new();

        // Structural validation
        if let Err(e) = self.structural_validator.validate_model_structure(model) {
            errors.push(e);
        }

        // Runtime validation for states and transitions
        let states = model.states();
        for state in states.iter() {
            if let Err(e) = self.runtime_validator.validate_state_activation(state) {
                errors.push(e);
            }
        }

        let transitions = model.transitions();
        for t in transitions {
            if let Err(e) = self.runtime_validator.validate_transition_consistency(t) {
                errors.push(e);
            }
        }

        // Rule-based validation
        {
            let rule_engine = self
                .rule_engine
                .lock()
                .map_err(|_| ValidationError::UnknownError("Poisoned mutex".into()))?;
            if let Err(e) = rule_engine.apply_rules(model) {
                errors.push(e);
            }
        }

        // Attempt recovery if errors are present and a recovery coordinator is provided
        if !errors.is_empty() {
            if let Some(recovery) = &self.recovery_coordinator {
                let mut recovery_errors = Vec::new();
                for e in &errors {
                    if let Err(recovery_err) = recovery.recover_from_error(model, e) {
                        recovery_errors.push(recovery_err);
                    }
                }
                errors.extend(recovery_errors);
            }
        }

        Ok(ValidationResult {
            valid: errors.is_empty(),
            errors,
        })
    }
}

/// A structural validator that ensures the model's structure is correct.
/// Checks that all transitions reference existing source and target states, etc.
#[derive(Debug, Default)]
pub struct FsmStructuralValidator<M: ModelDefinition> {
    _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelDefinition> StructuralValidator for FsmStructuralValidator<M> {
    type Model = M;
    type Error = ValidationError;

    fn validate_model_structure(&self, model: &Self::Model) -> Result<(), Self::Error> {
        let states = model.states();
        let transitions = model.transitions();

        let state_ids: Vec<_> = states.iter().map(|s| s.id()).collect();

        for t in transitions {
            let src = t.source_state_id();
            let tgt = t.target_state_id();
            if !state_ids.contains(&src) {
                return Err(ValidationError::StructuralError(format!(
                    "Missing source state: {}",
                    src
                )));
            }
            if !state_ids.contains(&tgt) {
                return Err(ValidationError::StructuralError(format!(
                    "Missing target state: {}",
                    tgt
                )));
            }
        }

        Ok(())
    }
}

/// A runtime validator that checks whether states and transitions are suitable for execution.
#[derive(Debug, Default)]
pub struct FsmRuntimeValidator<S: StateDefinition, T: TransitionDefinition> {
    _phantom: std::marker::PhantomData<(S, T)>,
}

impl<StateDef: StateDefinition, TransitionDef: TransitionDefinition> RuntimeValidator
    for FsmRuntimeValidator<StateDef, TransitionDef>
{
    type StateDef = StateDef;
    type TransitionDef = TransitionDef;
    type Error = ValidationError;

    fn validate_state_activation(&self, state: &Self::StateDef) -> Result<(), Self::Error> {
        // Ensure the state is well-defined for runtime activation.
        if state.id().trim().is_empty() {
            return Err(ValidationError::RuntimeError("State has empty ID".into()));
        }
        Ok(())
    }

    fn validate_transition_consistency(
        &self,
        transition: &Self::TransitionDef,
    ) -> Result<(), Self::Error> {
        // Ensure the transition is well-defined for runtime execution.
        if transition.event_id().trim().is_empty() {
            return Err(ValidationError::RuntimeError(format!(
                "Transition from {} to {} has an empty event_id.",
                transition.source_state_id(),
                transition.target_state_id()
            )));
        }
        Ok(())
    }
}

/// A rule engine that enforces domain-specific constraints on the model.
/// Users can register custom rules that must be satisfied.
#[derive(Default)]
pub struct FsmRuleEngine<M: ModelDefinition> {
    rules: HashMap<String, Box<dyn Fn(&M) -> Result<(), ValidationError> + Send + Sync>>,
}

impl<M: ModelDefinition> RuleEngine for FsmRuleEngine<M> {
    type Model = M;
    type Error = ValidationError;

    fn apply_rules(&self, model: &Self::Model) -> Result<(), Self::Error> {
        for (name, rule) in &self.rules {
            if let Err(e) = rule(model) {
                return Err(ValidationError::RuleError(format!(
                    "Rule '{}' failed: {:?}",
                    name, e
                )));
            }
        }
        Ok(())
    }

    fn register_rule(
        &mut self,
        rule_name: &str,
        rule_fn: Box<dyn Fn(&Self::Model) -> Result<(), Self::Error> + Send + Sync + 'static>,
    ) {
        self.rules.insert(rule_name.to_string(), rule_fn);
    }
}

/// A recovery coordinator that attempts to resolve detected validation errors.
#[derive(Debug, Default)]
pub struct FsmRecoveryCoordinator<M: ModelDefinition> {
    _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelDefinition> RecoveryCoordinator for FsmRecoveryCoordinator<M> {
    type Model = M;
    type Error = ValidationError;

    fn recover_from_error(
        &self,
        _model: &Self::Model,
        error: &Self::Error,
    ) -> Result<(), Self::Error> {
        // If recovery is not possible, return a suitable error.
        // In a real implementation, this might attempt corrective actions.
        Err(ValidationError::RecoveryError(format!(
            "Could not recover from error: {:?}",
            error
        )))
    }
}
