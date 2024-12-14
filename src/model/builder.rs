pub trait ModelBuilderInternal: Send {
    type Error;
    type StateDef: super::StateDefinition;
    type EventDef: super::EventDefinition;
    type TransitionDef: super::TransitionDefinition<
        StateDef = Self::StateDef,
        EventDef = Self::EventDef,
    >;

    fn define_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn define_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn define_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;
    fn finalize(
        self,
    ) -> Result<
        (
            Vec<Self::StateDef>,
            Vec<Self::EventDef>,
            Vec<Self::TransitionDef>,
        ),
        Self::Error,
    >;
}
