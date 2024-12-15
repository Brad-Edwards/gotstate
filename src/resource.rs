// SPDX-License-Identifier: MIT OR Apache-2.0

//! Provides traits for resource management.

/// `ResourceManager` manages external resources required by the FSM.
///
/// **Key Considerations:**
/// - Allocates and releases external resources (e.g., file handles, network connections).
/// - Ensures proper cleanup during system shutdown.
pub trait ResourceManager: Send + Sync {
    type ResourceHandle: Send + Sync;
    type ResourceType: Send + Sync;
    type ResourceConfig: Send + Sync;
    type Error;

    fn allocate_resource(
        &self,
        rtype: &Self::ResourceType,
        config: &Self::ResourceConfig,
    ) -> Result<Self::ResourceHandle, Self::Error>;
    fn release_resource(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}

/// `ResourcePool` handles the low-level resource allocation mechanism.
///
/// **Key Considerations:**
/// - Abstracts how resources are pooled and reused.
/// - Ensures that resource exhaustion and leaks are managed.
pub trait ResourcePool: Send + Sync {
    type ResourceHandle: Send + Sync;
    type ResourceType: Send + Sync;
    type ResourceConfig: Send + Sync;
    type Error;

    fn allocate(
        &self,
        rtype: &Self::ResourceType,
        config: &Self::ResourceConfig,
    ) -> Result<Self::ResourceHandle, Self::Error>;
    fn release(&self, handle: Self::ResourceHandle) -> Result<(), Self::Error>;
    fn cleanup(&self) -> Result<(), Self::Error>;
}

/// `ResourceTracker` keeps track of resource usage counts.
///
/// **Key Considerations:**
/// - Helps detect resource leaks or dangling references.
/// - Useful for debugging and optimizing resource utilization.
pub trait ResourceTracker: Send + Sync {
    type ResourceHandle: Send + Sync;

    fn increment_ref(&self, handle: &Self::ResourceHandle);
    fn decrement_ref(&self, handle: &Self::ResourceHandle);
    fn ref_count(&self, handle: &Self::ResourceHandle) -> usize;
}
