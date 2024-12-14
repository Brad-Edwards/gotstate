mod event_queue;
mod lock_manager;
mod timeout_scheduler;

pub use event_queue::EventQueue;
pub use lock_manager::LockManager;
pub use timeout_scheduler::TimeoutScheduler;

pub trait Concurrency: Send + Sync {
    type Event: Send;
    type LockHandle: Send;
    type Error;
    type Duration;

    fn enqueue_event(&self, event: Self::Event) -> Result<(), Self::Error>;
    fn dequeue_event(&self) -> Option<Self::Event>;
    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error>;
    fn release_lock(&self, handle: Self::LockHandle);
    fn schedule_timeout(
        &self,
        event: Self::Event,
        delay: Self::Duration,
    ) -> Result<(), Self::Error>;
    fn cancel_timeout(&self, event: &Self::Event) -> Result<(), Self::Error>;
}
