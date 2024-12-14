use crate::model::{EventDefinition, StateDefinition, TransitionDefinition};

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
