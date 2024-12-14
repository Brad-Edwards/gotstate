pub trait FSMBuilder: Send {
    type Error;
    type Model;
    type Validator;

    fn add_state(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_event(&mut self, name: &str) -> Result<(), Self::Error>;
    fn add_transition(&mut self, from: &str, to: &str, event: &str) -> Result<(), Self::Error>;
    fn build(self) -> Result<Self::Model, Self::Error>;
    fn validate_and_build(self, validator: &Self::Validator) -> Result<Self::Model, Self::Error>;
}
