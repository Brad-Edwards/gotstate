pub trait Validator: Send + Sync {
    type Model;
    type ValidationResult;
    type Error;

    fn validate_model(&self, model: &Self::Model) -> Result<Self::ValidationResult, Self::Error>;
}
