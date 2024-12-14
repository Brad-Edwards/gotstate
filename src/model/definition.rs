pub trait StateDefinition: Send + Sync {
    fn name(&self) -> &str;
    fn children(&self) -> &[Self]
    where
        Self: Sized;
    fn is_composite(&self) -> bool;
}

pub trait EventDefinition: Send + Sync {
    fn name(&self) -> &str;
}

pub trait TransitionDefinition: Send + Sync {
    type StateDef: StateDefinition;
    type EventDef: EventDefinition;

    fn source_state(&self) -> &Self::StateDef;
    fn target_state(&self) -> &Self::StateDef;
    fn event(&self) -> &Self::EventDef;
    fn has_guard(&self) -> bool;
}

pub trait ModelDefinition: Send + Sync {
    type StateDefinition: StateDefinition;
    type EventDefinition: EventDefinition;
    type TransitionDefinition: TransitionDefinition<
        StateDef = Self::StateDefinition,
        EventDef = Self::EventDefinition,
    >;

    fn root_state(&self) -> &Self::StateDefinition;
    fn states(&self) -> &[Self::StateDefinition];
    fn events(&self) -> &[Self::EventDefinition];
    fn transitions(&self) -> &[Self::TransitionDefinition];
}
