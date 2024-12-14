pub trait HierarchyValidator: Send + Sync {
    type StateDef: super::StateDefinition;
    type Error;

    fn validate_hierarchy(&self, root_state: &Self::StateDef) -> Result<(), Self::Error>;
}
