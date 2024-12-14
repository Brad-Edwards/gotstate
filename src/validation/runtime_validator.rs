pub trait RuntimeValidator: Send + Sync {
    type StateDef;
    type TransitionDef;
    type Error;

    fn validate_state_activation(&self, state: &Self::StateDef) -> Result<(), Self::Error>;
    fn validate_transition_consistency(
        &self,
        transition: &Self::TransitionDef,
    ) -> Result<(), Self::Error>;
}
