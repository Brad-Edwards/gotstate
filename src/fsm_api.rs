// SPDX-License-Identifier: MIT OR Apache-2.0

//! # FSM API
use crate::behavior::Behavior;
use crate::concurrency::Concurrency;
use crate::core::CoreEngine;
use crate::diagnostics::Diagnostics;
use crate::model::ModelDefinition;
use crate::resource::ResourceManager;
use crate::validator::*;

/// `FsmApi` provides a high-level API for creating and interacting with FSM instances.
///
/// **Key Considerations:**
/// - Simplifies integration into end-user applications.
/// - Ensures that all required components (model, concurrency, behavior, diagnostics, resources, validation) are provided.
pub trait FsmApi: Send + Sync {
    type Engine: CoreEngine;
    type Model: ModelDefinition;
    type Behavior: Behavior<TransitionDef = <Self::Model as ModelDefinition>::TransitionDefinition>;
    type Concurrency: Concurrency<Event = <Self::Engine as CoreEngine>::Event>;
    type Diagnostics: Diagnostics;
    type ResourceManager: ResourceManager;
    type Validator: Validator<Model = Self::Model>;
    type Error;

    /// Creates a new FSM instance from all provided components.
    /// This is typically where validation is applied and resources are initialized.
    fn create_fsm(
        &self,
        model: Self::Model,
        concurrency: Self::Concurrency,
        behavior: Self::Behavior,
        diagnostics: Self::Diagnostics,
        resource_manager: Self::ResourceManager,
        validator: Self::Validator,
    ) -> Result<Self::Engine, Self::Error>;

    /// Retrieves information about the current state of the given engine.
    fn get_state_info(&self, engine: &Self::Engine) -> <Self::Engine as CoreEngine>::StateHandle;

    /// Dispatches an event into the engine and returns the result of the transition.
    fn dispatch_event(
        &self,
        engine: &Self::Engine,
        event: <Self::Engine as CoreEngine>::Event,
    ) -> Result<<Self::Engine as CoreEngine>::TransitionResult, Self::Error>;
}
