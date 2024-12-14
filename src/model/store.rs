pub trait ImmutableModelStore: Send + Sync {
    type StateDef: super::StateDefinition;
    type EventDef: super::EventDefinition;
    type TransitionDef: super::TransitionDefinition<
        StateDef = Self::StateDef,
        EventDef = Self::EventDef,
    >;

    fn root_state(&self) -> &Self::StateDef;
    fn states(&self) -> &[Self::StateDef];
    fn events(&self) -> &[Self::EventDef];
    fn transitions(&self) -> &[Self::TransitionDef];
    fn find_state_by_name(&self, name: &str) -> Option<&Self::StateDef>;
    fn find_event_by_name(&self, name: &str) -> Option<&Self::EventDef>;
}
