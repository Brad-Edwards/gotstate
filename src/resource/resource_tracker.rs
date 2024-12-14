pub trait ResourceTracker: Send + Sync {
    type ResourceHandle: Send + Sync;

    fn increment_ref(&self, handle: &Self::ResourceHandle);
    fn decrement_ref(&self, handle: &Self::ResourceHandle);
    fn ref_count(&self, handle: &Self::ResourceHandle) -> usize;
}
