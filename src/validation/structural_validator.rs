pub trait StructuralValidator: Send + Sync {
    type Model;
    type Error;

    fn validate_model_structure(&self, model: &Self::Model) -> Result<(), Self::Error>;
}
