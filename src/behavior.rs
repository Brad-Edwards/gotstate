// SPDX-License-Identifier: MIT OR Apache-2.0

//! Behavior interfaces for state machine implementations.
use crate::diagnostics::Diagnostics;
use crate::model::{StateDefinition, TransitionDefinition};

/// `BehaviorContext` provides runtime information to behavior implementations.
///
/// **Key Considerations:**
/// - Allows actions and guards to access the current state, diagnostics, and resources.
/// - Promotes separation of pure model logic from contextual data.
pub trait BehaviorContext: Send + Sync {
    type StateHandle: Send + Sync;
    type Diagnostics: Diagnostics;
    type ResourceHandle: Send + Sync;

    fn current_state(&self) -> Self::StateHandle;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn resource_handle(&self) -> &Self::ResourceHandle;
}

/// `Behavior` defines a full set of behavior operations: guard evaluation, action execution,
/// and handling state entry/exit. This trait can stand alone or be composed from separate traits
/// (`GuardEvaluator`, `ActionExecutor`, `StateLifecycleHandler`). Implementations can mix and match
/// these patterns based on complexity and reusability needs.
pub trait Behavior: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context: BehaviorContext;

    fn evaluate_guard(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context,
    ) -> Result<bool, Self::Error>;
    fn execute_action(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
    fn on_state_entry(
        &self,
        state: &<Self::TransitionDef as TransitionDefinition>::StateDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
    fn on_state_exit(
        &self,
        state: &<Self::TransitionDef as TransitionDefinition>::StateDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
}

/// `GuardEvaluator` separates guard logic from action logic, allowing modular behavior definitions.
///
/// **Key Considerations:**
/// - Useful for testing guard logic in isolation.
/// - May be combined with `Behavior` or used as a standalone component.
pub trait GuardEvaluator: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context: BehaviorContext;

    fn evaluate_guard(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context,
    ) -> Result<bool, Self::Error>;
}

/// `ActionExecutor` handles the actual side-effect logic of a transition.
///
/// **Key Considerations:**
/// - Encourages separation of concerns by isolating actions from other logic.
pub trait ActionExecutor: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context: BehaviorContext;

    fn execute_action(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
}

/// `StateLifecycleHandler` manages state entry/exit behavior.
///
/// **Key Considerations:**
/// - Keeps state changes cleanly isolated from other behaviors.
/// - Useful for setting up resources or logging upon state change.
pub trait StateLifecycleHandler: Send + Sync {
    type Error;
    type StateDef: StateDefinition;
    type Context: BehaviorContext;

    fn on_state_entry(
        &self,
        state: &Self::StateDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
    fn on_state_exit(&self, state: &Self::StateDef, ctx: &Self::Context)
        -> Result<(), Self::Error>;
}
