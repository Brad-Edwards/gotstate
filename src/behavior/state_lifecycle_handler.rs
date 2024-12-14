use crate::model::StateDefinition;

pub trait StateLifecycleHandler: Send + Sync {
    type Error;
    type StateDef: StateDefinition;
    type Context;

    fn on_state_entry(
        &self,
        state: &Self::StateDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
    fn on_state_exit(&self, state: &Self::StateDef, ctx: &Self::Context)
        -> Result<(), Self::Error>;
}
