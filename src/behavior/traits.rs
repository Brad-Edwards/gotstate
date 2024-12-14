use crate::model::TransitionDefinition;

pub trait BehaviorContext: Send + Sync {
    type StateHandle: Send + Sync;
    type Diagnostics;
    type ResourceHandle: Send + Sync;

    fn current_state(&self) -> Self::StateHandle;
    fn diagnostics(&self) -> &Self::Diagnostics;
    fn resource_handle(&self) -> &Self::ResourceHandle;
}

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
