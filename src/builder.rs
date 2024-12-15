// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builder module provides a high-level interface for building and validating FSM models.
use crate::model::{EventDefinition, ModelDefinition, StateDefinition, TransitionDefinition};
use crate::validator::Validator;

/// FSMBuilder provides a high-level interface for building and validating FSM models.
///
/// **Key Considerations:**
/// - Simplifies model creation by abstracting internal builder steps.
/// - Validation can be deferred or performed immediately.
pub trait FSMBuilder: Send {
    type Error;
    type Model: ModelDefinition;
    type Validator: Validator<Model = Self::Model>;

    fn add_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;

    fn build(self) -> Result<Self::Model, Self::Error>;
    fn validate_and_build(self, validator: &Self::Validator) -> Result<Self::Model, Self::Error>;
}

/// BuilderContext is an internal abstraction for building states, events, and transitions.
///
/// **Key Considerations:**
/// - Typically a lower-level part of the builder system.
/// - Ensures a consistent final model.
pub trait BuilderContext: Send {
    type StateDefinition: StateDefinition;
    type EventDefinition: EventDefinition;
    type TransitionDefinition: TransitionDefinition<
        StateDef = Self::StateDefinition,
        EventDef = Self::EventDefinition,
    >;
    type Error;

    fn add_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;

    fn finalize(
        self,
    ) -> Result<
        (
            Vec<Self::StateDefinition>,
            Vec<Self::EventDefinition>,
            Vec<Self::TransitionDefinition>,
        ),
        Self::Error,
    >;
}
