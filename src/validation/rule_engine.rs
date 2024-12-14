pub trait RuleEngine: Send + Sync {
    type Model;
    type Error;

    fn apply_rules(&self, model: &Self::Model) -> Result<(), Self::Error>;
    fn register_rule(
        &mut self,
        rule_name: &str,
        rule_fn: &dyn Fn(&Self::Model) -> Result<(), Self::Error>,
    );
}

pub trait RecoveryCoordinator: Send + Sync {
    type Model;
    type Error;

    fn recover_from_error(
        &self,
        model: &Self::Model,
        error: &Self::Error,
    ) -> Result<(), Self::Error>;
}
