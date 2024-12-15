// SPDX-License-Identifier: MIT OR Apache-2.0

//! Provides traits for concurrency primitives.

/// `Concurrency` trait provides unified access to event queueing, locking, and timeouts.
///
/// **Key Considerations:**
/// - Ensures that multiple threads can safely enqueue/dequeue events.
/// - Provides a mechanism to schedule timeouts for delayed events.
/// - Acquires and releases locks to protect shared resources.
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

/// `EventQueue` provides a basic FIFO event handling interface.
///
/// **Key Considerations:**
/// - Allows the system to store incoming events until they can be processed.
/// - Must be thread-safe.
pub trait EventQueue: Send + Sync {
    type Event: Send;
    type Error;

    fn enqueue(&self, event: Self::Event) -> Result<(), Self::Error>;
    fn dequeue(&self) -> Option<Self::Event>;
    fn is_empty(&self) -> bool;
}

/// `LockManager` abstracts synchronization primitives.
///
/// **Key Considerations:**
/// - Ensures a uniform interface for locks across different platforms.
/// - Must prevent deadlocks and ensure timely releases.
pub trait LockManager: Send + Sync {
    type LockHandle: Send;
    type Error;

    fn acquire_lock(&self) -> Result<Self::LockHandle, Self::Error>;
    fn release_lock(&self, handle: Self::LockHandle);
}

/// `TimeoutScheduler` schedules and cancels timeouts for events.
///
/// **Key Considerations:**
/// - Useful for delayed state transitions.
/// - Must handle cancellation and expiration checks reliably.
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
