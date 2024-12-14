pub trait CoreEngine: Send + Sync {
    type Error;
    type StateHandle: Send + Sync;
    type Event: Send;
    type TransitionResult: Send;

    fn start(&self) -> Result<(), Self::Error>;
    fn stop(&self) -> Result<(), Self::Error>;
    fn process_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;
    fn current_state(&self) -> Self::StateHandle;
}
