pub trait CoreDispatcher: Send + Sync {
    type Error;
    type Event: Send;
    type TransitionResult: Send;
    type Executor;

    fn dispatch_event(&self, event: Self::Event) -> Result<Self::TransitionResult, Self::Error>;
    fn process_queued_events(&self) -> Result<(), Self::Error>;
}
