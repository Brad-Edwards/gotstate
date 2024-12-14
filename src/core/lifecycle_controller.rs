pub trait LifecycleController: Send + Sync {
    type Error;

    fn start(&self) -> Result<(), Self::Error>;
    fn stop(&self) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}
