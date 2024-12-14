use crate::model::TransitionDefinition;

pub trait ActionExecutor: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context;

    fn execute_action(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context,
    ) -> Result<(), Self::Error>;
}
