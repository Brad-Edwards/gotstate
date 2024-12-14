pub trait TimeoutScheduler: Send + Sync {
    type Event: Send;
    type Duration;
    type Error;

    fn schedule_timeout(
        &self,
        event: Self::Event,
        delay: Self::Duration,
    ) -> Result<(), Self::Error>;
    fn cancel_timeout(&self, event: &Self::Event) -> Result<(), Self::Error>;
    fn check_expired(&self) -> Result<Vec<Self::Event>, Self::Error>;
}
