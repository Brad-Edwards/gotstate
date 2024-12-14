use crate::model::TransitionDefinition;

pub trait GuardEvaluator: Send + Sync {
    type Error;
    type TransitionDef: TransitionDefinition;
    type Context;

    fn evaluate_guard(
        &self,
        transition: &Self::TransitionDef,
        ctx: &Self::Context,
    ) -> Result<bool, Self::Error>;
}
